"""Shared task semantics; no inference imports or implicit model switching."""
EDIT_SCOPE='Apply only the requested changes to the source photo. Preserve all other content, identity, framing and appearance unless the requested edit requires changing them.'


def supported_tasks(profile):
    return ['generate','edit'] if profile.get('reference_mode')=='edit' else ['generate']


GRID_NAMES=(('top-left corner','top edge','top-right corner'),('left side','center','right side'),('bottom-left corner','bottom edge','bottom-right corner'))


def describe_mask_region(mask):
    """Name where a white-edits mask lies on a 3x3 grid, since the model never sees the mask itself."""
    width,height=mask.size
    binary=mask.point(lambda value:255 if value>127 else 0)
    coverage={}
    for row in range(3):
        for column in range(3):
            cell=binary.crop((column*width//3,row*height//3,(column+1)*width//3,(row+1)*height//3))
            coverage[row,column]=cell.histogram()[255]/(cell.width*cell.height)
    selected=[cell for cell,share in coverage.items() if share>=0.4]
    if not selected:
        # Thin strokes and small spots fill no cell, so name every cell they pass through.
        touched=[cell for cell,share in coverage.items() if share>=0.005] or [max(coverage,key=coverage.get)]
        box=binary.getbbox()
        if box is None:
            return 'a small part of the image'
        filled=binary.histogram()[255]/((box[2]-box[0])*(box[3]-box[1]))
        names=', '.join(GRID_NAMES[r][c] for r,c in touched)
        return f'a thin strip running through the {names} of the image' if filled<0.3 else f'a small area in the {names} of the image'
    if len(selected)==9:
        return 'the whole image'
    border=[cell for cell in selected if cell!=(1,1)]
    if (1,1) not in selected and len(border)>=5:
        return 'the background around the main subject, along the '+', '.join(GRID_NAMES[r][c] for r,c in border)
    return 'the '+', '.join(GRID_NAMES[r][c] for r,c in selected)+' of the image'


def mask_boxes(mask, grid=64, max_boxes=8):
    """Bounding boxes on Qwen3-VL's 0-1000 grounding scale, one per painted area.

    Long thin areas (strokes along an edge) are split into pieces along their length,
    so the boxes follow the stroke instead of claiming everything inside its curve.
    """
    small=mask.convert('L').resize((grid,grid)).point(lambda value:255 if value>40 else 0)
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
