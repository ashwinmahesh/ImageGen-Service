"""Selected-area location hints, patch widening and the complete edit-prompt budget, with no model weights/GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw
import photo_workflows
import prompt_expander
from photo_workflows import (EDIT_SCOPE, MAX_LOCATION_BOXES, MAX_PROMPT_CHARACTERS, PROMPT_TOO_LONG, WORST_CASE_REGION,
                             describe_mask_boxes, edit_prompt, edit_prompt_fits, mask_boxes)
from reference_geometry import prepare_geometry, prepare_mask, widen_mask_to_grid

ROOT=Path(__file__).resolve().parents[1]


def painted(size, *shapes, value=255):
    """A black square mask with filled rectangles given as (left, top, right, bottom) pixel boxes."""
    mask=Image.new('L',(size,size),0)
    draw=ImageDraw.Draw(mask)
    for shape in shapes:
        draw.rectangle(shape,fill=value)
    return mask


def inside(box, x, y, size):
    """Whether the centre of pixel (x, y) lies in a 0-1000 grounding box."""
    cx,cy=(x+.5)*1000/size,(y+.5)*1000/size
    return box[0]<=cx<=box[2] and box[1]<=cy<=box[3]


def covered(boxes, x, y, size):
    return any(inside(box,x,y,size) for box in boxes)


class MaskBoxesTest(unittest.TestCase):
    def test_separate_regions_become_separate_boxes_largest_first(self):
        mask=painted(1024,(100,100,299,299),(700,100,799,199),(700,700,759,759))
        boxes=mask_boxes(mask)
        self.assertEqual(len(boxes),3)
        for box,(x,y) in zip(boxes,[(200,200),(750,150),(730,730)]):
            self.assertTrue(inside(box,x,y,1024),(box,x,y))
        self.assertFalse(covered(boxes,500,500,1024))

    def test_box_count_is_capped_at_the_eight_largest_regions(self):
        sizes=[160,144,128,112,96,80,64,48,10,10,10,10]
        centres=[]
        shapes=[]
        for index,length in enumerate(sizes):
            left,top=20+250*(index%4),20+250*(index//4)
            shapes.append((left,top,left+length-1,top+length-1))
            centres.append((left+length//2,top+length//2))
        boxes=mask_boxes(painted(1024,*shapes))
        self.assertEqual(len(boxes),MAX_LOCATION_BOXES)
        for x,y in centres[:8]:
            self.assertTrue(covered(boxes,x,y,1024),(x,y))
        for x,y in centres[8:]:
            self.assertFalse(covered(boxes,x,y,1024),(x,y))

    def test_thin_curve_splits_into_boxes_that_follow_it(self):
        mask=Image.new('L',(1024,1024),0)
        ImageDraw.Draw(mask).arc((60,60,960,960),90,270,fill=255,width=12)
        boxes=mask_boxes(mask)
        self.assertTrue(2<=len(boxes)<=4,boxes)
        # One box around the whole arc would also claim the untouched space inside the curve.
        self.assertFalse(covered(boxes,500,500,1024))
        self.assertTrue(covered(boxes,66,510,1024))
        self.assertTrue(covered(boxes,510,66,1024))
        self.assertTrue(covered(boxes,510,954,1024))

    def test_minimum_brush_dots_and_one_pixel_strokes_keep_a_hint(self):
        for size in (512,1024,2048):
            diameter=max(1,round(size*.01))
            centre=size//2
            with self.subTest(size=size,shape='minimum brush dot'):
                # Centred on the image, so the dot straddles four grid cells and averaging would dilute it.
                radius=diameter//2
                mask=Image.new('L',(size,size),0)
                ImageDraw.Draw(mask).ellipse((centre-radius,centre-radius,centre+radius,centre+radius),fill=255)
                boxes=mask_boxes(mask)
                self.assertEqual(len(boxes),1)
                self.assertTrue(inside(boxes[0],centre,centre,size))
                self.assertTrue(describe_mask_boxes(mask).startswith('{"bbox_2d": ['))
            for name,line in (('horizontal',(size//4,centre,3*size//4,centre)),('vertical',(centre,size//4,centre,3*size//4)),
                              ('diagonal',(size//4,size//4,3*size//4,3*size//4))):
                with self.subTest(size=size,shape=name+' one-pixel stroke'):
                    mask=Image.new('L',(size,size),0)
                    ImageDraw.Draw(mask).line(line,fill=255,width=1)
                    boxes=mask_boxes(mask)
                    self.assertTrue(boxes)
                    self.assertTrue(covered(boxes,line[0],line[1],size))
                    self.assertTrue(covered(boxes,line[2],line[3],size))
            with self.subTest(size=size,shape='single pixels at the corners'):
                for corner in ((0,0),(size-1,0),(0,size-1),(size-1,size-1)):
                    mask=Image.new('L',(size,size),0)
                    mask.putpixel(corner,1)
                    boxes=mask_boxes(mask)
                    self.assertEqual(len(boxes),1)
                    self.assertTrue(inside(boxes[0],*corner,size))

    def test_every_nonzero_grey_level_is_selected_and_black_areas_are_not(self):
        for value in (1,64,127,128,255):
            with self.subTest(value=value):
                boxes=mask_boxes(painted(1024,(100,100,199,199),value=value))
                self.assertEqual(len(boxes),1)
                self.assertTrue(inside(boxes[0],150,150,1024))
                self.assertTrue(boxes[0][0]<=100/1.024<=boxes[0][0]+16 and boxes[0][2]>=199/1.024)
                # Only the touched 1/64th cells are claimed, not the untouched black area around them.
                self.assertLessEqual(boxes[0][2],199/1.024+16)
                self.assertLessEqual(boxes[0][3],199/1.024+16)
                self.assertFalse(covered(boxes,800,800,1024))
                self.assertFalse(covered(boxes,50,50,1024))

    def test_empty_mask_has_no_boxes_or_hint(self):
        empty=Image.new('L',(1024,1024),0)
        self.assertEqual(mask_boxes(empty),[])
        self.assertIsNone(describe_mask_boxes(empty))

    def test_non_square_and_tiny_masks_do_not_lose_selected_pixels(self):
        wide=Image.new('L',(1024,128),0)
        wide.putpixel((700,64),255)
        self.assertEqual(len(mask_boxes(wide)),1)
        tiny=Image.new('L',(10,7),0)
        tiny.putpixel((9,6),255)
        boxes=mask_boxes(tiny)
        self.assertEqual(len(boxes),1)
        self.assertTrue(inside(boxes[0],9,6,10) or boxes[0][2]==1000)

    def test_hint_text_uses_the_qwen_grounding_format_and_sits_before_the_scope(self):
        region=describe_mask_boxes(painted(1024,(0,656,431,1023)))
        self.assertEqual(region,'{"bbox_2d": [0, 641, 422, 1000]}')
        prompt=edit_prompt('fill it with flowers','keep the face',region)
        lines=prompt.split('\n')
        self.assertEqual(lines[0],'fill it with flowers')
        self.assertEqual(lines[1],'The selected area to edit is inside these boxes (x1, y1, x2, y2 on a 0-1000 scale of the image): '
                         +region+'. Make the requested change only there.')
        self.assertEqual(lines[2],EDIT_SCOPE+' Explicit preservation requirements: keep the face')
        self.assertEqual(edit_prompt('fill it with flowers','keep the face'),
                         'fill it with flowers\n'+EDIT_SCOPE+' Explicit preservation requirements: keep the face')

    def test_unused_region_description_helpers_are_gone(self):
        self.assertFalse(hasattr(photo_workflows,'describe_mask_region'))
        self.assertFalse(hasattr(photo_workflows,'GRID_NAMES'))


class PreparedMaskCoordinatesTest(unittest.TestCase):
    def test_hint_is_in_the_padded_image_the_model_sees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source,mask=root/'source.png',root/'mask.png'
            Image.new('RGB',(1024,128),'red').save(source)
            painted_mask=Image.new('L',(1024,128),0)
            ImageDraw.Draw(painted_mask).rectangle((128,96,255,127),fill=255)
            painted_mask.save(mask)
            with prepare_geometry(str(source),'auto',root/'scratch') as geometry:
                self.assertEqual(geometry.model_size,(1024,256))
                with prepare_mask(str(mask),geometry,root/'scratch') as prepared:
                    with Image.open(prepared) as runtime_mask:
                        self.assertEqual(runtime_mask.size,(1024,256))
                        self.assertEqual(describe_mask_boxes(runtime_mask),'{"bbox_2d": [125, 625, 250, 750]}')
            # The original photo's coordinates would have pointed into the bottom padding.
            self.assertEqual(describe_mask_boxes(painted_mask),'{"bbox_2d": [125, 750, 250, 1000]}')


class WidenMaskTest(unittest.TestCase):
    def widened(self, mask, name='mask.png'):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path=Path(directory.name)/name
        mask.save(path)
        output=Path(widen_mask_to_grid(str(path)))
        self.assertEqual(output.name,'model-mask.png')
        self.assertNotEqual(output,path)
        with Image.open(output) as image:
            return image.convert('L'),path

    def test_single_white_pixel_becomes_its_whole_patch(self):
        mask=Image.new('L',(64,64),0)
        mask.putpixel((20,37),255)
        result,_=self.widened(mask)
        self.assertEqual(result.size,(64,64))
        for y in range(64):
            for x in range(64):
                self.assertEqual(result.getpixel((x,y)),255 if 16<=x<32 and 32<=y<48 else 0,(x,y))

    def test_values_are_kept_not_thresholded_or_averaged(self):
        for value in (1,64,127,128,255):
            with self.subTest(value=value):
                mask=Image.new('L',(64,64),0)
                mask.putpixel((20,20),value)
                result,_=self.widened(mask)
                self.assertEqual(result.getextrema(),(0,value))
                self.assertEqual(result.crop((16,16,32,32)).getextrema(),(value,value))

    def test_soft_region_survives_in_its_model_patches(self):
        mask=painted(256,(32,32,63,63),value=64)
        result,_=self.widened(mask)
        self.assertEqual(result.crop((32,32,64,64)).getextrema(),(64,64))
        self.assertEqual(result.histogram()[64],32*32)
        self.assertEqual(result.histogram()[0],256*256-32*32)

    def test_patch_takes_the_strongest_value_it_contains(self):
        mask=Image.new('L',(32,16),0)
        mask.putpixel((1,1),10)
        mask.putpixel((14,9),200)
        mask.putpixel((20,3),30)
        result,_=self.widened(mask)
        self.assertEqual(result.crop((0,0,16,16)).getextrema(),(200,200))
        self.assertEqual(result.crop((16,0,32,16)).getextrema(),(30,30))

    def test_partial_edge_patches_keep_the_input_size(self):
        mask=Image.new('L',(50,40),0)
        mask.putpixel((49,39),255)
        result,_=self.widened(mask)
        self.assertEqual(result.size,(50,40))
        for y in range(40):
            for x in range(50):
                self.assertEqual(result.getpixel((x,y)),255 if x>=48 and y>=32 else 0,(x,y))
        single=Image.new('L',(17,17),0)
        single.putpixel((16,16),255)
        result,_=self.widened(single)
        self.assertEqual(result.size,(17,17))
        self.assertEqual(result.histogram()[255],1)

    def test_original_mask_file_is_unchanged(self):
        mask=Image.new('L',(64,48),0)
        mask.putpixel((5,5),255)
        mask.putpixel((40,30),90)
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path=Path(directory.name)/'mask.png'
        mask.save(path)
        before=path.read_bytes()
        widen_mask_to_grid(str(path))
        self.assertEqual(path.read_bytes(),before)
        with Image.open(path) as image:
            self.assertEqual(image.getpixel((6,5)),0)
            self.assertEqual(image.getpixel((40,30)),90)


def refine(prompt, details, **changes):
    """Run the edit refiner worker with a canned LLM reply."""
    response=types.SimpleNamespace(text=json.dumps(details),finish_reason='stop',peak_memory=.1)
    class Tokenizer:
        def apply_chat_template(self,value,**kwargs):
            return 'formatted edit'
    lm=types.ModuleType('mlx_lm');lm.load=lambda _: (object(),Tokenizer());lm.stream_generate=lambda *a,**k:iter([response])
    samples=types.ModuleType('mlx_lm.sample_utils');samples.make_sampler=lambda **kwargs: None
    with patch.dict(sys.modules,{'mlx_lm':lm,'mlx_lm.sample_utils':samples}):
        return prompt_expander.run_worker({'prompt':prompt,'weights':'unused','image_model':'qwen-image-2.1-turbo','task':'edit',**changes})


def eight_box_region():
    shapes=[(16+48*column,40+110*row,23+48*column,47+110*row) for column in range(4) for row in range(2)]
    region=describe_mask_boxes(painted(256,*shapes))
    assert region.count('bbox_2d')==MAX_LOCATION_BOXES
    return region


PRESERVE='Keep the face, pose and background unchanged.'


def sized_instruction(total, preserve=PRESERVE, region=None):
    """An instruction making the complete rendering prompt exactly `total` characters."""
    return 'a'*(total-len(edit_prompt('',preserve,region)))


class CompletePromptBudgetTest(unittest.TestCase):
    def test_limit_is_checked_on_the_complete_prompt_with_eight_boxes(self):
        region=eight_box_region()
        for total,fits in ((MAX_PROMPT_CHARACTERS,True),(MAX_PROMPT_CHARACTERS+1,False)):
            with self.subTest(total=total):
                instruction=sized_instruction(total,region=region)
                self.assertEqual(len(edit_prompt(instruction,PRESERVE,region)),total)
                self.assertIs(edit_prompt_fits(instruction,PRESERVE,region=region),fits)

    def test_worst_case_reservation_covers_any_eight_boxes(self):
        self.assertEqual(WORST_CASE_REGION.count('bbox_2d'),MAX_LOCATION_BOXES)
        self.assertGreaterEqual(len(WORST_CASE_REGION),len(eight_box_region()))
        instruction=sized_instruction(MAX_PROMPT_CHARACTERS,region=WORST_CASE_REGION)
        self.assertTrue(edit_prompt_fits(instruction,PRESERVE,masked=True))
        self.assertFalse(edit_prompt_fits(instruction+'a',PRESERVE,masked=True))

    def test_request_validation_cannot_see_the_hint_so_it_counts_instruction_and_scope_only(self):
        instruction=sized_instruction(MAX_PROMPT_CHARACTERS)
        self.assertTrue(edit_prompt_fits(instruction,PRESERVE))
        self.assertFalse(edit_prompt_fits(instruction+'a',PRESERVE))
        # The same instruction needs the authoritative check once boxes exist.
        self.assertFalse(edit_prompt_fits(instruction,PRESERVE,region=eight_box_region()))

    def test_refiner_reserves_worst_case_hint_room_for_masked_edits(self):
        original=('Change the jacket to green. '*108).strip()
        for masked in (True,False):
            hint=WORST_CASE_REGION if masked else None
            base=len(edit_prompt(original+'\n',PRESERVE,hint))
            for total,accepted in ((MAX_PROMPT_CHARACTERS,True),(MAX_PROMPT_CHARACTERS+1,False)):
                with self.subTest(masked=masked,total=total):
                    details={'edit_details':'x'*(total-base),'preservation':''}
                    request={'preserve':PRESERVE,**({'masked':True} if masked else {})}
                    if accepted:
                        data=refine(original,details,**request)
                        self.assertEqual(len(edit_prompt(data['expanded_prompt'],PRESERVE,hint)),total)
                    else:
                        with self.assertRaisesRegex(prompt_expander.ExpansionError,'final prompt budget'):
                            refine(original,details,**request)

    def test_refinement_that_fits_without_hints_fails_for_masked_edits(self):
        original=('Change the jacket to green. '*108).strip()
        details={'edit_details':'x'*(MAX_PROMPT_CHARACTERS-len(edit_prompt(original+'\n',PRESERVE))),'preservation':''}
        data=refine(original,details,preserve=PRESERVE)
        self.assertEqual(len(edit_prompt(data['expanded_prompt'],PRESERVE)),MAX_PROMPT_CHARACTERS)
        with self.assertRaisesRegex(prompt_expander.ExpansionError,'selected-area guidance'):
            refine(original,details,preserve=PRESERVE,masked=True)

    def test_refiner_never_truncates_the_instruction_or_preservation(self):
        original='Change the jacket to green.'
        data=refine(original,{'edit_details':'Change only the jacket color.','preservation':'Keep the face.'},preserve=PRESERVE,masked=True)
        self.assertTrue(data['expanded_prompt'].startswith(original+'\n'))
        self.assertIn('Keep the face.',data['expanded_prompt'])
        with self.assertRaises(prompt_expander.ExpansionError):
            refine(original,{'edit_details':'Change only the jacket color.','preservation':''},preserve=PRESERVE,masked='yes')

    def test_masked_flag_reaches_the_refiner_only_for_selected_area_edits(self):
        requests=[]
        def run(command,**kwargs):
            request=json.loads(kwargs['input']);requests.append(request)
            output=json.dumps({'original_prompt':request['prompt'],'expanded_prompt':request['prompt']+'\nKeep other details.','task':'edit'})
            return subprocess.CompletedProcess(command,0,stdout=output,stderr='')
        with patch.object(prompt_expander,'check_assets',return_value=Path('/unused')),patch.object(prompt_expander.subprocess,'run',side_effect=run):
            for kwargs in ({'task':'edit','preserve':PRESERVE,'masked':True},{'task':'edit','preserve':PRESERVE},{}):
                prompt_expander.expand('Change the jacket to green.',project=ROOT,image_model='qwen-image-2.1-turbo',image_weights='unused',**kwargs)
        self.assertIs(requests[0]['masked'],True)
        self.assertNotIn('masked',requests[1])
        self.assertNotIn('masked',requests[2])
        self.assertNotIn('task',requests[2])


class MaskedEditIntegrationTest(unittest.TestCase):
    """Fake inference engine behind the real HTTP routes, queue and compositing. Qwen alone accepts masks."""

    def run_script(self, script):
        with tempfile.TemporaryDirectory() as temporary:
            result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,
                env={**os.environ,'IMAGE_MODEL':'qwen-image-2.1-turbo','DATA_DIR':temporary},
                capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)

    def test_grounding_inference_and_compositing_use_the_right_masks(self):
        self.run_script(r'''
from pathlib import Path
from PIL import Image,ImageDraw
from unittest.mock import patch
import server
from photo_workflows import EDIT_SCOPE
from tests.test_api import APITest
fixture=APITest();fixture.setUp()
try:
    inference=[];composites=[]
    real_generate=fixture.fake.generate_image
    def spy_generate(**kwargs):
        entry={'mask_name':None}
        if kwargs.get('mask_path'):
            entry['mask_name']=Path(kwargs['mask_path']).name
            with Image.open(kwargs['mask_path']) as image: entry['mask']=image.convert('L'); entry['mask'].load()
        inference.append(entry)
        return real_generate(**kwargs)
    fixture.fake.generate_image=spy_generate
    real_composite=server.composite_edit
    def spy_composite(path,source_path,mask_path):
        with Image.open(mask_path) as image: mask=image.convert('L'); mask.load()
        composites.append({'mask_name':Path(mask_path).name,'mask':mask})
        return real_composite(path,source_path,mask_path)
    # A 2x2 pixel selection inside one 16px model patch of a 256x256 photo, over a photo with no flat areas.
    source=fixture.root/'source.png'
    photo=Image.new('RGB',(256,256))
    photo.putdata([(x,y,128) for y in range(256) for x in range(256)]); photo.save(source)
    mask=fixture.root/'mask.png'
    pixels=Image.new('L',(256,256),0); ImageDraw.Draw(pixels).rectangle((50,50,51,51),fill=255); pixels.save(mask)
    request={'reference_image':str(source),'mask_image':str(mask),'prompt':'Make it red.','preserve':'face and background',
             'seed':3,'output_dir':str(fixture.root/'out'),'response_format':'url'}
    with patch.object(server,'composite_edit',side_effect=spy_composite):
        response=fixture.client.post('/v1/images/edits',json=request)
    assert response.status_code==200,response.text
    exact=fixture.fake.calls[-1]['prompt']
    # Boxes come from the exact prepared mask (cell 12 of 64), not from the widened 16px patch (cells 12-15).
    expected=('Make it red.\nThe selected area to edit is inside these boxes (x1, y1, x2, y2 on a 0-1000 scale of the image): '
              '{"bbox_2d": [188, 188, 203, 203]}. Make the requested change only there.\n'
              +EDIT_SCOPE+' Explicit preservation requirements: face and background')
    assert exact==expected,exact
    assert len(inference)==1 and inference[0]['mask_name']=='model-mask.png',inference
    widened=inference[0]['mask']
    assert widened.getpixel((48,48))==255 and widened.getpixel((63,63))==255,'inference must get the widened patch'
    assert widened.getpixel((47,50))==0 and widened.getpixel((64,50))==0
    assert len(composites)==1 and composites[0]['mask_name']=='mask.png',composites
    original=composites[0]['mask']
    assert original.histogram()[255]==4 and original.getpixel((50,50))==255 and original.getpixel((52,52))==0,'compositing must use the exact mask'
    assert Path(fixture.fake.calls[-1]['mask_path']).name=='model-mask.png'
    with Image.open(response.json()['data'][0]['file_path']) as output:
        output=output.convert('RGBA')
        assert output.size==(256,256)
        assert output.getpixel((50,50))==(0,128,0,255) and output.getpixel((51,51))==(0,128,0,255)
        for y in range(256):
            for x in range(256):
                if not (50<=x<=51 and 50<=y<=51):
                    assert output.getpixel((x,y))==(x,y,128,255),(x,y,output.getpixel((x,y)))
    assert fixture.fake.calls[-1]['image_path']==str(source)
    assert Path(source).exists() and Path(mask).exists()
    print('OK')
finally:fixture.tearDown()
''')

    def test_padded_runtime_coordinates_reach_the_prompt(self):
        self.run_script(r'''
from PIL import Image,ImageDraw
import server
from tests.test_api import APITest
fixture=APITest();fixture.setUp()
try:
    source=fixture.root/'wide.png'; Image.new('RGB',(1024,128),'red').save(source)
    mask=fixture.root/'wide-mask.png'
    pixels=Image.new('L',(1024,128),0); ImageDraw.Draw(pixels).rectangle((128,96,255,127),fill=255); pixels.save(mask)
    # Auto resolution is the default for reference edits, so the 1024x128 photo is padded to 1024x256.
    response=fixture.client.post('/v1/images/edits',json={'reference_image':str(source),'mask_image':str(mask),'prompt':'Make it blue.',
        'seed':3,'output_dir':str(fixture.root/'out'),'response_format':'url'})
    assert response.status_code==200,response.text
    call=fixture.fake.calls[-1]
    assert (call['width'],call['height'])==(1024,256),call
    assert '{"bbox_2d": [125, 625, 250, 750]}' in call['prompt'],call['prompt']
    assert '[125, 750, 250, 1000]' not in call['prompt']
    with Image.open(response.json()['data'][0]['file_path']) as output:
        assert output.size==(1024,128)
        assert output.convert('RGB').getpixel((200,110))==(0,128,0)
        assert output.convert('RGB').getpixel((200,50))==(255,0,0)
    print('OK')
finally:fixture.tearDown()
''')

    def test_unmasked_edits_and_plain_generation_keep_their_prompts_and_inputs(self):
        self.run_script(r'''
from PIL import Image
import server
from photo_workflows import EDIT_SCOPE
from tests.test_api import APITest
fixture=APITest();fixture.setUp()
try:
    source=fixture.root/'source.png'; Image.new('RGB',(256,256),'red').save(source)
    response=fixture.client.post('/v1/images/edits',json={**fixture.payload,'reference_image':str(source),'prompt':'Change the jacket to green.','preserve':'face and background'})
    assert response.status_code==200,response.text
    call=fixture.fake.calls[-1]
    assert call['prompt']=='Change the jacket to green.\n'+EDIT_SCOPE+' Explicit preservation requirements: face and background',call['prompt']
    assert 'bbox_2d' not in call['prompt'] and 'mask_path' not in call and call['image_path']==str(source)
    response=fixture.client.post('/v1/images/edits',json={**fixture.payload,'reference_image':str(source),'prompt':'Change the jacket to green.'})
    assert response.status_code==200,response.text
    assert fixture.fake.calls[-1]['prompt']=='Change the jacket to green.\n'+EDIT_SCOPE
    response=fixture.post(prompt='A landscape photograph')
    assert response.status_code==200,response.text
    call=fixture.fake.calls[-1]
    assert call['prompt']=='A landscape photograph' and call['image_path'] is None and 'mask_path' not in call
    print('OK')
finally:fixture.tearDown()
''')

    # Shared setup for the prompt-limit scenarios: a 256x256 photo with eight separate painted squares.
    LIMIT_SETUP=r"""
import json
from pathlib import Path
from unittest.mock import patch
from PIL import Image,ImageDraw
import server,prompt_expander
from photo_workflows import PROMPT_TOO_LONG,edit_prompt,describe_mask_boxes
from tests.test_api import APITest
fixture=APITest();fixture.setUp()
try:
    source=fixture.root/'source.png'; Image.new('RGB',(256,256),'red').save(source)
    mask=fixture.root/'mask.png'
    pixels=Image.new('L',(256,256),0); draw=ImageDraw.Draw(pixels)
    for column in range(4):
        for row in range(2): draw.rectangle((16+48*column,40+110*row,23+48*column,47+110*row),fill=255)
    pixels.save(mask)
    region=describe_mask_boxes(pixels); assert region.count('bbox_2d')==8
    preserve='Keep the face, pose and background unchanged.'
    def sized(total): return 'a'*(total-len(edit_prompt('',preserve,region)))
    out=fixture.root/'out'
    request={'reference_image':str(source),'mask_image':str(mask),'preserve':preserve,'seed':3,'output_dir':str(out),'response_format':'url'}
    plain={key:value for key,value in request.items() if key!='mask_image'}
"""

    def test_complete_prompt_limit_with_eight_boxes_without_refinement(self):
        self.run_script(self.LIMIT_SETUP+r"""
    response=fixture.client.post('/v1/images/edits',json={**request,'prompt':sized(4000)})
    assert response.status_code==200,response.text
    prompt=fixture.fake.calls[-1]['prompt']
    assert len(prompt)==4000 and prompt.count('bbox_2d')==8 and preserve in prompt and prompt.startswith(sized(4000)+'\n')
    calls=len(fixture.fake.calls)
    response=fixture.client.post('/v1/images/edits',json={**request,'prompt':sized(4001)})
    assert response.status_code==400 and response.json()['detail']==PROMPT_TOO_LONG,(response.status_code,response.text)
    assert len(fixture.fake.calls)==calls,'the engine must not run for an over-budget prompt'
    assert not list(out.glob('.*working.png')) and len(list(out.glob('*.png')))==1,list(out.iterdir())
    # Unmasked edits keep the request validator's exact check: no hint is added to their prompt.
    assert fixture.client.post('/v1/images/edits',json={**plain,'prompt':'a'*(4000-len(edit_prompt('',preserve)))}).status_code==200
    assert fixture.client.post('/v1/images/edits',json={**plain,'prompt':'a'*(4001-len(edit_prompt('',preserve)))}).status_code==422
    print('OK')
finally:fixture.tearDown()
""")

    def test_queue_admission_accepts_an_over_budget_mask_job_and_consumption_rejects_it(self):
        self.run_script(self.LIMIT_SETUP+r"""
    fixture.worker.stop(); fixture.job_executor.shutdown(wait=True)
    for origin in ('generator','queue'):
        body={**request,'task':'edit','prompt':sized(4001)}
        client=fixture.client if origin=='generator' else fixture.queue_http
        accepted=client.post('/v1/jobs',json=body if origin=='generator' else {'model':server.API_MODEL,'request':body},headers={'Idempotency-Key':'long-'+origin})
        assert accepted.status_code==202,accepted.text
        claimed=fixture.store.claim()
        try: server.execute_job(claimed)
        except server.LocalFileError as error: assert str(error)==PROMPT_TOO_LONG
        else: raise AssertionError('expected the complete prompt check to fail')
        fixture.store.finish(claimed['id'],error=PROMPT_TOO_LONG)
    assert fixture.fake.calls==[]
    print('OK')
finally:fixture.tearDown()
""")

    def test_complete_prompt_limit_with_eight_boxes_after_refinement(self):
        self.run_script(self.LIMIT_SETUP+r"""
    fixture.worker.llm_enabled=lambda:True
    refined=[]
    def refiner(prompt,**kwargs):
        refined.append(kwargs)
        # The refiner only sees the instruction; the boxes are added afterwards from the prepared mask.
        return dict(original_prompt=prompt,expanded_prompt=prompt+'\n'+sized(target[0])[len(prompt)+1:],task='edit',
                    model='fake',revision='pinned',mode='expand',policy_version=1,seconds=.1)
    target=[4000]
    with patch.object(server.app.state,'llm_enabled',True),patch.object(prompt_expander,'expand',side_effect=refiner):
        response=fixture.client.post('/v1/images/edits',json={**request,'prompt':'Make it red.','expand_prompt':True})
        assert response.status_code==200,response.text
        assert refined[-1]['masked'] is True and refined[-1]['task']=='edit' and refined[-1]['preserve']==preserve,refined[-1]
        prompt=fixture.fake.calls[-1]['prompt']
        assert len(prompt)==4000 and prompt.count('bbox_2d')==8 and preserve in prompt and prompt.startswith('Make it red.\n')
        calls=len(fixture.fake.calls)
        target[0]=4001
        response=fixture.client.post('/v1/images/edits',json={**request,'prompt':'Make it red.','expand_prompt':True})
        assert response.status_code==400 and response.json()['detail']==PROMPT_TOO_LONG,(response.status_code,response.text)
        assert len(fixture.fake.calls)==calls
        # Unmasked refinement is called with exactly its previous arguments.
        target[0]=1200
        response=fixture.client.post('/v1/images/edits',json={**plain,'prompt':'Make it red.','expand_prompt':True})
        assert response.status_code==200,response.text
        assert set(refined[-1])=={'task','preserve','project','image_model','image_weights','selection','inherited_fds'}.intersection(refined[-1]) and 'masked' not in refined[-1],refined[-1]
    print('OK')
finally:fixture.tearDown()
""")


if __name__=='__main__':
    unittest.main()
