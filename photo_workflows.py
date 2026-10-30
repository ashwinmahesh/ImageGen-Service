"""Shared task semantics; no inference imports or implicit model switching."""
EDIT_SCOPE='Apply only the requested changes to the source photo. Preserve all other content, identity, framing and appearance unless the requested edit requires changing them.'
MAX_PROMPT_CHARACTERS=4000
MAX_LOCATION_BOXES=8
# Longest possible location hint: the maximum number of boxes with four-digit coordinates.
WORST_CASE_REGION=', '.join(['{"bbox_2d": [1000, 1000, 1000, 1000]}']*MAX_LOCATION_BOXES)
PROMPT_TOO_LONG=(f'The complete edit prompt exceeds {MAX_PROMPT_CHARACTERS} characters, including preservation and selected-area guidance. '
                 'Shorten the instruction or preservation text, or disable refinement.')


def supported_tasks(profile):
    return ['generate','edit'] if profile.get('reference_mode')=='edit' else ['generate']


def mask_boxes(mask, grid=64, max_boxes=MAX_LOCATION_BOXES):
    """Bounding boxes on Qwen3-VL's 0-1000 grounding scale, one per painted area.

    A grid cell counts as painted when any pixel in its footprint is nonzero, so a minimum brush dot,
    a one-pixel stroke or a faint grey selection still yields a box. Long thin areas (strokes along an
    edge) are split into pieces along their length, so the boxes follow the stroke instead of claiming
    everything inside its curve.
    """
    from PIL import Image
    gray=mask.convert('L')
    full_width,full_height=gray.size
    small=Image.new('L',(grid,grid))
    small.putdata([
        255 if gray.crop((x*full_width//grid,y*full_height//grid,((x+1)*full_width+grid-1)//grid,((y+1)*full_height+grid-1)//grid)).getextrema()[1] else 0
        for y in range(grid) for x in range(grid)])
    pixels=small.load()
    painted={(x,y) for y in range(grid) for x in range(grid) if pixels[x,y]}
    areas=[]
    while painted:
        stack=[painted.pop()]; area=[]
        while stack:
            x,y=stack.pop(); area.append((x,y))
            for neighbour in ((x+1,y),(x-1,y),(x,y+1),(x,y-1),(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)):
                if neighbour in painted:
                    painted.remove(neighbour); stack.append(neighbour)
        areas.append(area)
    areas.sort(key=len,reverse=True)
    boxes=[]
    for area in areas:
        xs=[x for x,_ in area]; ys=[y for _,y in area]
        width,height=max(xs)-min(xs)+1,max(ys)-min(ys)+1
        pieces=1 if len(area)>=0.3*width*height else min(4,max(2,max(width,height)//12))
        axis=0 if width>=height else 1
        area.sort(key=lambda cell:cell[axis])
        for index in range(pieces):
            piece=area[index*len(area)//pieces:(index+1)*len(area)//pieces]
            px=[x for x,_ in piece]; py=[y for _,y in piece]
            boxes.append([round(min(px)*1000/grid),round(min(py)*1000/grid),round((max(px)+1)*1000/grid),round((max(py)+1)*1000/grid)])
    return boxes[:max_boxes]


def describe_mask_boxes(mask):
    boxes=mask_boxes(mask)
    return ', '.join('{"bbox_2d": [%d, %d, %d, %d]}'%tuple(box) for box in boxes) if boxes else None


def edit_prompt(instruction, preserve=None, region=None):
    scope=EDIT_SCOPE+(' Explicit preservation requirements: '+preserve.strip() if preserve else '')
    if region:
        scope=f'The selected area to edit is inside these boxes (x1, y1, x2, y2 on a 0-1000 scale of the image): {region}. Make the requested change only there.\n'+scope
    return instruction+'\n'+scope


def edit_prompt_fits(instruction, preserve=None, *, region=None, masked=False):
    """Whether the complete rendering prompt stays within the limit.

    Pass the real ``region`` once the boxes are known. ``masked`` instead reserves worst-case hint
    room when they are not (before refinement); without either, the hint is not counted, which is
    the lower bound that request validation can check without reading the photo or mask.
    """
    if region is None and masked:
        region=WORST_CASE_REGION
    return len(edit_prompt(instruction,preserve,region))<=MAX_PROMPT_CHARACTERS
