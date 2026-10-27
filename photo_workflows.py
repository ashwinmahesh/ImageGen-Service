"""Shared task semantics; no inference imports or implicit model switching."""
EDIT_SCOPE='Apply only the requested changes to the source photo. Preserve all other content, identity, framing and appearance unless the requested edit requires changing them.'


def supported_tasks(profile):
    return ['generate','edit'] if profile.get('reference_mode')=='edit' else ['generate']


GRID_NAMES=(('top-left corner','top edge','top-right corner'),('left side','center','right side'),('bottom-left corner','bottom edge','bottom-right corner'))


def describe_mask_region(mask):
    """Name where a white-edits mask lies on a 3x3 grid, since the model never sees the mask itself."""
    width,height=mask.size
    selected=[]
    for row in range(3):
        for column in range(3):
            cell=mask.crop((column*width//3,row*height//3,(column+1)*width//3,(row+1)*height//3))
            if sum(cell.point(lambda value:255 if value>127 else 0).histogram()[255:])>=0.4*cell.width*cell.height:
                selected.append((row,column))
    if not selected:
        return 'a small part of the image'
    if len(selected)==9:
        return 'the whole image'
    border=[cell for cell in selected if cell!=(1,1)]
    if (1,1) not in selected and len(border)>=5:
        return 'the background around the main subject, along the '+', '.join(GRID_NAMES[r][c] for r,c in border)
    return 'the '+', '.join(GRID_NAMES[r][c] for r,c in selected)+' of the image'


def edit_prompt(instruction, preserve=None, region=None):
    scope=EDIT_SCOPE+(' Explicit preservation requirements: '+preserve.strip() if preserve else '')
    if region:
        scope=f'The selected area to edit is {region}. Make the requested change there.\n'+scope
    return instruction+'\n'+scope
