/* Help buttons never submit forms or toggle neighboring controls. */
(() => {
  const tooltip = document.createElement('div');
  tooltip.id = 'setting-help-tooltip';
  tooltip.className = 'help-tooltip';
  tooltip.setAttribute('role', 'tooltip');
  tooltip.hidden = true;
  document.body.append(tooltip);
  let active = null, closeTimer;

  function hide() {
    clearTimeout(closeTimer);
    tooltip.hidden = true;
    if (active) active.removeAttribute('aria-describedby');
    active = null;
  }
  function show(button) {
    hide();
    active = button;
    // A modal's children stay above its backdrop, including this shared tooltip.
    (button.closest('dialog') || document.body).append(tooltip);
    tooltip.textContent = button.dataset.help;
    tooltip.hidden = false;
    button.setAttribute('aria-describedby', tooltip.id);
    const bounds = button.getBoundingClientRect();
    const box = tooltip.getBoundingClientRect();
    const left = Math.max(12, Math.min(bounds.left, innerWidth - box.width - 12));
    let top = bounds.bottom + 8;
    if (top + box.height > innerHeight - 12) top = bounds.top - box.height - 8;
    top = Math.max(12, Math.min(top, innerHeight - box.height - 12));
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
  }
  function closeSoon() {
    clearTimeout(closeTimer);
    closeTimer = setTimeout(() => {
      if (active && document.activeElement !== active && !active.matches(':hover') && !tooltip.matches(':hover')) hide();
    }, 120);
  }
  for (const button of document.querySelectorAll('button[data-help]')) {
    button.addEventListener('mouseenter', () => show(button));
    button.addEventListener('mouseleave', closeSoon);
    button.addEventListener('focus', () => show(button));
    button.addEventListener('blur', closeSoon);
    button.addEventListener('click', event => {
      event.stopPropagation();
      show(button);
    });
  }
  tooltip.addEventListener('mouseenter', () => clearTimeout(closeTimer));
  tooltip.addEventListener('mouseleave', closeSoon);
  document.addEventListener('pointerdown', event => {
    if (active && event.target !== active && !active.contains(event.target) && !tooltip.contains(event.target)) hide();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && active) {
      hide();
      event.preventDefault();
      event.stopPropagation();
    }
  }, true);
  document.addEventListener('scroll', hide, true);
  document.addEventListener('close', hide, true);
  window.addEventListener('resize', hide);
})();
