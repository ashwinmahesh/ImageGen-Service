"""Run an offline, short-lived LLM child while the caller owns the GPU lease."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from enhancer_profile import REPOSITORY, REVISION, check_assets

import hashlib
import re
from prompt_policy import POLICY_VERSION, SYSTEM_PROMPT, VARIATION_SYSTEM_PROMPT, VARIATION_GOALS, EXAMPLE_USER, EXAMPLE_ASSISTANT, SAMPLING, guidance_for, public_guidance, style_context, style_anchor, example_for
from prompt_policy import EDIT_POLICY_VERSION, EDIT_SYSTEM_PROMPT, EDIT_EXAMPLE_USER, EDIT_EXAMPLE_ASSISTANT, edit_guidance_for
from model_profiles import PROFILES
from photo_workflows import supported_tasks, edit_prompt



class ExpansionError(Exception):
    http_status = 422


def expansion_capability(enabled, image_model=None):
    return {'enabled': enabled, 'mode': 'expand', 'model': REPOSITORY, 'revision': REVISION,
            'license': 'Apache-2.0', 'text_only': True,
            'guidance_profile': guidance_for(image_model)['id'] if image_model else None,
            'batch_variations': True,
            'photo_editing':{'available':image_model in PROFILES and 'edit' in supported_tasks(PROFILES[image_model]),
                            'guidance_profile':edit_guidance_for(image_model)['id'] if image_model in PROFILES and 'edit' in supported_tasks(PROFILES[image_model]) else None,
                            'text_only':True,'composition_variation':False},
            'note': 'Keeps your description verbatim and appends visual detail. Does not inspect reference photos. Opt in with expand_prompt:true; add vary_prompt:true for distinct batch compositions. No automatic compression.'}


def expand(prompt, *, project, image_model, image_weights, selection=None, inherited_fds=(), task='generate', preserve=None):
    return _expand(prompt,project=project,image_model=image_model,image_weights=image_weights,
                   selection=selection,inherited_fds=inherited_fds,task=task,preserve=preserve)


def expand_batch(prompt, *, count, project, image_model, image_weights, selection=None, inherited_fds=()):
    return _expand(prompt,count=count,project=project,image_model=image_model,image_weights=image_weights,
                   selection=selection,inherited_fds=inherited_fds)


def _expand(prompt, *, project, image_model, image_weights, selection=None, inherited_fds=(), count=None, task='generate', preserve=None):
    if task not in ('generate','edit'):
        raise ValueError('Unknown expansion task')
    folder = check_assets(project)
    command = [sys.executable, str(Path(__file__).with_name('prompt_expander.py')), '--worker']
    request = {'prompt': prompt, 'weights': str(folder), 'image_model': image_model,
               'image_weights': str(image_weights), 'trigger': (selection or {}).get('trigger_prompt', '')}
    if task=='edit':
        if count is not None:
            raise ValueError('Photo edits cannot use composition variation')
        request.update(task='edit',preserve=preserve)
    if count is not None:
        if type(count) is not int or not 2<=count<=5: raise ValueError('Variation count must be 2–5')
        request['variation_count']=count
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'TOKENIZERS_PARALLELISM': 'false'}
    started = time.monotonic()
    try:
        result = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True,
                                env=environment, pass_fds=tuple(inherited_fds), timeout=180)
    except subprocess.TimeoutExpired as error:
        raise ExpansionError('Prompt expansion exceeded 180 seconds. No image was rendered; retry or disable expansion.') from error
    if result.returncode:
        # stdout is our structured protocol; stderr may contain private prompt/runtime traces.
        try:
            message = json.loads(result.stdout)['error']
        except (ValueError, KeyError):
            message = 'Local prompt expander failed. Check its installation, or disable expansion.'
        raise ExpansionError(message)
    try:
        data = json.loads(result.stdout)
        if count is None: validate_expansion(prompt, data)
        else: validate_variations(prompt,data['candidates'],count)
    except (ValueError, KeyError, TypeError) as error:
        raise ExpansionError('Prompt expander returned an invalid response; no image was rendered.') from error
    if count is not None:
        return [{**item,'model':REPOSITORY,'revision':REVISION,'mode':'expand','policy_version':POLICY_VERSION}
                for item in data['candidates']]
    return {**data, 'model': REPOSITORY, 'revision': REVISION, 'mode': 'expand',
            'policy_version': EDIT_POLICY_VERSION if task=='edit' else POLICY_VERSION, 'seconds': round(time.monotonic()-started, 2)}


def validate_expansion(original, data):
    expanded = data['expanded_prompt']
    if data['original_prompt'] != original or not expanded.startswith(original+'\n'):
        raise ValueError('Original description must be retained verbatim')
    if not expanded[len(original):].strip() or len(expanded) > 4000 or '<lora:' in expanded.lower():
        raise ValueError('Invalid expansion')


def validate_variations(original, candidates, count):
    if not isinstance(candidates,list) or len(candidates)!=count:
        raise ExpansionError('Wrong number of candidate expansions')
    for index,item in enumerate(candidates):
        try: validate_expansion(original,item)
        except (ValueError,KeyError,TypeError) as error:
            raise ExpansionError('Invalid candidate expansion; no new image was rendered.') from error
        if item.get('variation_index')!=index+1 or item.get('variation_count')!=count:
            raise ExpansionError('Wrong candidate expansion order')
    if len({item['expanded_prompt'] for item in candidates})!=count:
        raise ExpansionError('Candidate expansions repeated; no images were rendered. Try a less constrained prompt or disable composition variation.')


def parse_details(text, original):
    text = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text,re.S)
    if fenced: text = fenced.group(1)
    try:
        details = json.loads(text)
        if not isinstance(details,dict) or set(details) != {'style_details','additional_composition'}:
            raise ValueError('Wrong fields')
        for key,value in details.items():
            if not isinstance(value,str): raise ValueError('Detail fields must be strings')
            value = value.strip()
            # Remove only an exact input echo, never any of the authoritative original.
            if value.startswith(original): value = value[len(original):].strip()
            if '<think>' in value or '</think>' in value: raise ValueError('Reasoning in output')
            details[key] = value
        if not details['style_details'] and not details['additional_composition']: raise ValueError('No added detail')
        return {key:details[key] for key in ('style_details','additional_composition')}
    except (ValueError,TypeError) as error:
        raise ExpansionError('Prompt expansion did not return usable structured visual details; preview again or disable expansion.') from error


def check_added_details(original, addition):
    """Narrow regression guards, not a claim of general semantic verification."""
    if addition.startswith(('```', '# ', '- ')) or '<lora:' in addition.lower():
        raise ExpansionError('Expansion returned formatting/instructions instead of descriptive sentences.')
    rear = re.search(r'(?:seen|viewed|shown)\s+(?:strictly\s+)?from behind|(?:strict\s+)?rear[- ]view|back[- ]view|faces?\s+(?:never visible|hidden|invisible)',original,re.I)
    flat = re.search(r'flat.*(?:cel|vector|color)|(?:cel|vector).*flat',original,re.I)
    for sentence in re.split(r'(?<=[.!?])\s+',addition):
        negative = re.search(r'\b(?:no|not|never|without|hidden|unseen|invisible|concealed|obscured)\b',sentence,re.I)
        if rear and not negative and re.search(r'\b(?:faces?|eyes?|smiles?|facial expressions?)\b',sentence,re.I):
            raise ExpansionError('Expansion introduced facial detail into a rear-view scene; preview again or disable expansion.')
        if flat and not negative and re.search(r'photorealis|skin pores|fabric (?:weave|microtexture)|bokeh|lens flare|photographic|physically based',sentence,re.I):
            raise ExpansionError('Expansion introduced photographic detail into flat art; preview again or disable expansion.')


def image_token_count(prompt, request):
    if request['image_model'] != 'z-image-turbo':
        return None  # Other pipelines' formatting is not measured here; never substitute the LLM tokenizer.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path(request['image_weights'])/'tokenizer'), local_files_only=True)
    trigger = request.get('trigger', '').strip()
    from lora_catalog import effective_prompt
    effective = effective_prompt(prompt, {'trigger_prompt':trigger})
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': effective}],
                      tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return len(tokenizer.encode(formatted, add_special_tokens=True))


def run_worker(request):
    original = request['prompt']
    if not original.strip() or len(original) > 4000:
        raise ExpansionError('Prompt must contain 1–4000 characters.')
    if request.get('task')=='edit':
        return run_edit_worker(request)
    original_tokens = image_token_count(original, request)
    if original_tokens is not None and original_tokens >= 490:
        raise ExpansionError(f'Original prompt uses {original_tokens}/512 image-model tokens. There is too little room to expand; shorten it yourself or disable expansion. It was not compressed.')
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    model, tokenizer = load(request['weights'])
    guidance = guidance_for(request['image_model'])
    allowance = min(guidance['soft_max_new_words'], max(12, (480-original_tokens)//2)) if original_tokens is not None else guidance['soft_max_new_words']
    count=request.get('variation_count')
    if count is not None and (type(count) is not int or not 2<=count<=5):
        raise ExpansionError('Variation count must be 2–5')
    candidates=[]
    for index in range(count or 1):
        started=time.monotonic()
        system=VARIATION_SYSTEM_PROMPT if count else SYSTEM_PROMPT
        messages = [{'role': 'system', 'content': system+'\n\n'+guidance['instructions']+'\n\n'+style_context(original)}]
        example = example_for(original) if not count else None
        if example:
            messages += [{'role':'user','content':example[0]},{'role':'assistant','content':example[1]}]
        direction=''
        if count:
            direction=f'Option {index+1} of {count}. Composition goal: {VARIATION_GOALS[index]}\n'
            if candidates:
                direction+='Earlier compositions to differ from: '+json.dumps([item['details']['additional_composition'] for item in candidates])+'\n'
        messages.append({'role': 'user', 'content': direction+f'Expand this image description with at most {allowance} NEW words. Return the two-field JSON with new rendering detail:\n\n{original}'})
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text, final = '', None
        if SAMPLING['temperature']:
            import mlx.core as mx
            mx.random.seed(SAMPLING['seed'])
        for response in stream_generate(model, tokenizer, prompt=formatted, max_tokens=320,
              sampler=make_sampler(temp=SAMPLING['temperature'],top_p=SAMPLING['top_p'],top_k=SAMPLING['top_k']),
              logits_processors=make_logits_processors(presence_penalty=SAMPLING['presence_penalty'],presence_context_size=512)):
            text += response.text
            final = response
        if final is None or final.finish_reason == 'length':
            raise ExpansionError('Expansion reached its output limit; retry or disable expansion. No truncated text was used.')
        details = parse_details(text, original)
        anchor = style_anchor(original)
        if not anchor:
            details['style_details'] = ''
            if not details['additional_composition']:
                raise ExpansionError('No additional scene detail was returned for this unspecified medium.')
        if count and not details['additional_composition']:
            raise ExpansionError('No distinct composition was returned. Try a less constrained prompt or disable composition variation.')
        addition = ' '.join(details[key] for key in ('style_details','additional_composition') if details[key])
        check_added_details(original, addition)
        data = {'original_prompt': original, 'expanded_prompt': original+'\n'+addition,
                'original_image_tokens': original_tokens, 'image_token_limit': 512 if original_tokens is not None else None}
        validate_expansion(original, data)
        data.update(style_anchor=anchor,details=details,image_model=request['image_model'],
                    guidance_profile=guidance['id'],guidance_version=guidance['version'],sampling=dict(SAMPLING))
        data['policy_sha256'] = hashlib.sha256(json.dumps([system, guidance, style_context(original), example, direction],sort_keys=True).encode()).hexdigest()
        import resource
        data['peak_mlx_memory_gb'] = round(final.peak_memory, 3)
        data['peak_process_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == 'darwin' else 1e6), 3)
        data['expanded_image_tokens'] = image_token_count(data['expanded_prompt'], request)
        if data['expanded_image_tokens'] is not None and data['expanded_image_tokens'] > 512:
            raise ExpansionError(f"Expanded prompt uses {data['expanded_image_tokens']}/512 image-model tokens. It was not truncated or compressed; shorten your description or disable expansion.")
        if count:
            data.update(variation_index=index+1,variation_count=count,variation_goal=VARIATION_GOALS[index],seconds=round(time.monotonic()-started,2))
        candidates.append(data)
    if count:
        validate_variations(original,candidates,count)
        return {'candidates':candidates}
    return candidates[0]


def parse_edit_details(text, original):
    """Normalize one known field synonym without accepting extra edit fields."""
    try:
        value=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text.strip(),re.S)
        details=json.loads(value.group(1) if value else text)
        # The stock model sometimes echoes the public request's preserve key.
        if isinstance(details,dict) and set(details)=={'edit_details','preserve'}:
            details['preservation']=details.pop('preserve')
        if not isinstance(details,dict) or set(details)!={'edit_details','preservation'}:
            raise ValueError('expected edit_details and preservation strings')
        if not all(isinstance(item,str) for item in details.values()):
            raise ValueError('edit fields must be strings')
        details={key:value.strip() for key,value in details.items()}
        addition=' '.join(value for value in details.values() if value)
        if not addition or len(addition)>1200 or re.search(r'</?think>|<lora:',addition,re.I):
            raise ValueError('empty, overlong or improperly formatted edit details')
    except (ValueError,TypeError,KeyError) as error:
        raise ExpansionError('The refiner returned invalid edit JSON; retry or disable Prompt expansion.') from error
    # Keep framing guards, but do not mistake "Do not crop the image" for an edit.
    for sentence in re.split(r'(?<=[.!?])\s+',addition):
        for match in re.finditer(r'wide shot|close-up|new camera angle|reframe|crop the image',sentence,re.I):
            if match.group().lower() in original.lower():
                continue
            before=sentence[:match.start()]
            negated=re.search(r"(?:\b(?:do not|don't|never|avoid)\s+(?:(?:use|add|introduce|apply|create|switch to|change to)\s+)?|\b(?:without|no)\s+)(?:a\s+|an\s+|the\s+)?$",before,re.I)
            retained=re.search(r'\b(?:keep|retain|preserve|maintain)\s+(?:the\s+)?(?:original|existing|current)\s+$',before,re.I)
            if not (negated or retained):
                raise ExpansionError('The refiner introduced an unrequested framing change; retry or disable Prompt expansion.')
    return details,addition


def run_edit_worker(request):
    """Photo edits use their own policy, with no scene/composition expansion path."""
    original=request['prompt']
    identifier=request['image_model']
    if identifier not in PROFILES or 'edit' not in supported_tasks(PROFILES[identifier]):
        raise ExpansionError('Instruction refinement requires an editing-capable model')
    if request.get('variation_count') is not None:
        raise ExpansionError('Composition variation is unavailable for photo editing')
    preserve=request.get('preserve')
    if preserve is not None and (not isinstance(preserve,str) or len(preserve)>1000):
        raise ExpansionError('Preservation requirements must be at most 1000 characters')
    guidance=edit_guidance_for(identifier)
    messages=[{'role':'system','content':EDIT_SYSTEM_PROMPT+'\n\n'+guidance['instructions']},
              {'role':'user','content':EDIT_EXAMPLE_USER},
              {'role':'assistant','content':EDIT_EXAMPLE_ASSISTANT},
              {'role':'user','content':'Refine only this requested photo edit with at most 60 NEW words.\n'
               +json.dumps({'instruction':original,'preservation':preserve or ''},ensure_ascii=False)}]
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model,tokenizer=load(request['weights'])
    formatted=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    text,final='',None
    for response in stream_generate(model,tokenizer,prompt=formatted,max_tokens=256,sampler=make_sampler(temp=0)):
        text+=response.text
        final=response
    if final is None or final.finish_reason=='length':
        raise ExpansionError('Edit-instruction refinement exceeded its output limit; retry or disable refinement')
    details,addition=parse_edit_details(text,original)
    data={'original_prompt':original,'expanded_prompt':original+'\n'+addition,'task':'edit'}
    try:validate_expansion(original,data)
    except (ValueError,TypeError,KeyError) as error:
        raise ExpansionError('The refined edit exceeds its prompt budget; shorten the instruction or disable Prompt expansion.') from error
    if len(edit_prompt(data['expanded_prompt'],preserve))>4000:
        raise ExpansionError('The refined edit and preservation requirements exceed the final prompt budget; shorten them or disable Prompt expansion.')
    import resource
    data.update(details=details,image_model=identifier,guidance_profile=guidance['id'],guidance_version=guidance['version'],
                original_image_tokens=None,expanded_image_tokens=None,image_token_limit=None,
                sampling={'temperature':0.0},policy_version=EDIT_POLICY_VERSION,
                policy_sha256=hashlib.sha256(json.dumps([EDIT_SYSTEM_PROMPT,guidance,EDIT_EXAMPLE_USER,EDIT_EXAMPLE_ASSISTANT,preserve],sort_keys=True).encode()).hexdigest(),
                peak_mlx_memory_gb=round(final.peak_memory,3),
                peak_process_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e9 if sys.platform=='darwin' else 1e6),3))
    return data


if __name__ == '__main__':
    try:
        # Native library chatter belongs on stderr, not in the JSON protocol.
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            result = run_worker(json.load(sys.stdin))
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({'error': str(error) if isinstance(error, ExpansionError) else 'Local prompt runtime failed; check assets/runtime compatibility.'}))
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
