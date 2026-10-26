"""Shared task semantics; no inference imports or implicit model switching."""
EDIT_SCOPE='Apply only the requested changes to the source photo. Preserve all other content, identity, framing and appearance unless the requested edit requires changing them.'


def supported_tasks(profile):
    return ['generate','edit'] if profile.get('reference_mode')=='edit' else ['generate']


def edit_prompt(instruction, preserve=None):
    scope=EDIT_SCOPE+(' Explicit preservation requirements: '+preserve.strip() if preserve else '')
    return instruction+'\n'+scope
