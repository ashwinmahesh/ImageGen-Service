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
from prompt_policy import POLICY_VERSION, SYSTEM_PROMPT, EXAMPLE_USER, EXAMPLE_ASSISTANT, SAMPLING, guidance_for, public_guidance, style_context, style_anchor, example_for



class ExpansionError(Exception):
    http_status = 422


def expansion_capability(enabled, image_model=None):
    return {'enabled': enabled, 'mode': 'expand', 'model': REPOSITORY, 'revision': REVISION,
            'license': 'Apache-2.0', 'text_only': True,
            'guidance_profile': guidance_for(image_model)['id'] if image_model else None,
            'note': 'Keeps your description verbatim and appends visual detail. Does not inspect reference photos. Opt in with expand_prompt:true; no automatic compression.'}


def expand(prompt, *, project, image_model, image_weights, selection=None, inherited_fds=()):
    folder = check_assets(project)
    command = [sys.executable, str(Path(__file__).with_name('prompt_expander.py')), '--worker']
    request = {'prompt': prompt, 'weights': str(folder), 'image_model': image_model,
               'image_weights': str(image_weights), 'trigger': (selection or {}).get('trigger_prompt', '')}
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
        validate_expansion(prompt, data)
    except (ValueError, KeyError, TypeError) as error:
        raise ExpansionError('Prompt expander returned an invalid response; no image was rendered.') from error
    return {**data, 'model': REPOSITORY, 'revision': REVISION, 'mode': 'expand',
            'policy_version': POLICY_VERSION, 'seconds': round(time.monotonic()-started, 2)}


def validate_expansion(original, data):
    expanded = data['expanded_prompt']
    if data['original_prompt'] != original or not expanded.startswith(original+'\n'):
        raise ValueError('Original description must be retained verbatim')
    if not expanded[len(original):].strip() or len(expanded) > 4000 or '<lora:' in expanded.lower():
        raise ValueError('Invalid expansion')


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
        return None  # Native Chroma uses GGUF T5; do not substitute the LLM tokenizer.
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
    original_tokens = image_token_count(original, request)
    if original_tokens is not None and original_tokens >= 490:
        raise ExpansionError(f'Original prompt uses {original_tokens}/512 image-model tokens. There is too little room to expand; shorten it yourself or disable expansion. It was not compressed.')
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    model, tokenizer = load(request['weights'])
    guidance = guidance_for(request['image_model'])
    allowance = min(guidance['soft_max_new_words'], max(12, (480-original_tokens)//2)) if original_tokens is not None else guidance['soft_max_new_words']
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT+'\n\n'+guidance['instructions']+'\n\n'+style_context(original)}]
    # Use only an example appropriate to the requested medium, avoiding cross-style bias.
    example = example_for(original)
    if example:
        messages += [{'role':'user','content':example[0]},{'role':'assistant','content':example[1]}]
    messages.append({'role': 'user', 'content': f'Expand this image description with at most {allowance} NEW words. Return the two-field JSON with new rendering detail:\n\n{original}'})
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
    addition = ' '.join(details[key] for key in ('style_details','additional_composition') if details[key])
    check_added_details(original, addition)
    data = {'original_prompt': original, 'expanded_prompt': original+'\n'+addition,
            'original_image_tokens': original_tokens, 'image_token_limit': 512 if original_tokens is not None else None}
    validate_expansion(original, data)
    data['style_anchor'] = anchor
    data['details'] = details
    data['image_model'] = request['image_model']
    data['guidance_profile'] = guidance['id']
    data['guidance_version'] = guidance['version']
    data['sampling'] = dict(SAMPLING)
    data['policy_sha256'] = hashlib.sha256(json.dumps([SYSTEM_PROMPT, guidance, style_context(original), example],sort_keys=True).encode()).hexdigest()
    import resource
    data['peak_mlx_memory_gb'] = round(final.peak_memory, 3)
    data['peak_process_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == 'darwin' else 1e6), 3)
    data['expanded_image_tokens'] = image_token_count(data['expanded_prompt'], request)
    if data['expanded_image_tokens'] is not None and data['expanded_image_tokens'] > 512:
        raise ExpansionError(f"Expanded prompt uses {data['expanded_image_tokens']}/512 image-model tokens. It was not truncated or compressed; shorten your description or disable expansion.")
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
