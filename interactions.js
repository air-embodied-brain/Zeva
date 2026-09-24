(() => {
  'use strict';
  const hero = document.querySelector('.hero-head');
  if (!hero) return;
  hero.id ||= 'top';

  // Group related sections into six stable reading destinations.
  const sections = [...document.querySelectorAll('section.section')];
  const analysis = sections.find(section => section.querySelector('.case-studies'));
  if (analysis) analysis.id ||= 'analysis';
  const destinations = [
    ['overview-video', 'Overview'], ['videos', 'Demos'], ['method', 'Method'],
    ['capability-scaling', 'Results'], ['analysis', 'Analysis'], ['bibtex', 'BibTeX']
  ].filter(([id]) => document.getElementById(id));
  const nav = document.createElement('nav');
  nav.className = 'page-nav';
  nav.setAttribute('aria-label', 'Page sections');
  nav.innerHTML = `<div class="page-nav-inner"><a class="page-nav-brand" href="#top" aria-label="Zeva — back to top">Zeva</a><button class="ui-button nav-toggle" type="button" aria-expanded="false" aria-controls="page-nav-links">Sections</button><div class="page-nav-links" id="page-nav-links">${destinations.map(([id, label]) => `<a href="#${id}">${label}</a>`).join('')}</div></div>`;
  hero.after(nav);
  const toggle = nav.querySelector('button');
  const links = nav.querySelector('.page-nav-links');
  const closeMenu = () => { toggle.setAttribute('aria-expanded', 'false'); links.classList.remove('is-open'); };
  toggle.addEventListener('click', () => {
    const open = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(open));
    links.classList.toggle('is-open', open);
  });
  nav.addEventListener('keydown', event => { if (event.key === 'Escape') { closeMenu(); toggle.focus(); } });
  nav.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (!link) return;
    closeMenu();
    const target = document.querySelector(link.getAttribute('href'));
    target.setAttribute('tabindex', '-1');
    target.focus({ preventScroll: true });
  });
  const markers = destinations.map(([id]) => document.getElementById(id));
  let scheduled = false;
  const updateNavigation = () => {
    scheduled = false;
    let active = -1;
    markers.forEach((section, index) => { if (section.getBoundingClientRect().top <= 160) active = index; });
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) active = markers.length - 1;
    links.querySelectorAll('a').forEach((link, index) => {
      if (index === active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  };
  window.addEventListener('scroll', () => { if (!scheduled) { scheduled = true; requestAnimationFrame(updateNavigation); } }, { passive: true });
  window.addEventListener('resize', updateNavigation, { passive: true });
  updateNavigation();

  // Native dialog provides modal focus containment and Escape dismissal.
  const viewer = document.createElement('dialog');
  viewer.className = 'image-viewer';
  viewer.setAttribute('aria-labelledby', 'viewer-title');
  viewer.innerHTML = '<div class="viewer-toolbar"><h2 id="viewer-title">Image detail</h2><button type="button" class="ui-button" data-close autofocus aria-label="Close image viewer">Close ×</button></div><div class="viewer-images"></div><div class="viewer-footer"><button type="button" class="ui-button" data-prev aria-label="Previous image group">← Previous</button><p aria-live="polite"></p><button type="button" class="ui-button" data-next aria-label="Next image group">Next →</button></div>';
  document.body.append(viewer);
  let groups = [], position = 0, opener;
  const render = () => {
    const images = groups[position];
    const content = viewer.querySelector('.viewer-images');
    content.replaceChildren();
    content.style.setProperty('--columns', images.length);
    images.forEach(original => {
      const figure = document.createElement('figure');
      const img = document.createElement('img');
      img.src = original.currentSrc || original.src;
      img.alt = original.alt;
      const caption = document.createElement('figcaption');
      const state = original.closest('figure')?.querySelector('figcaption')?.textContent.trim();
      caption.textContent = state ? `${original.alt} — ${state}` : original.alt;
      figure.append(img, caption);
      content.append(figure);
    });
    const original = images[0];
    const title = original.closest('.effect-row, .case-study, .warmup-block, section')?.querySelector('h3, h2')?.textContent.trim() || 'Image detail';
    const kind = original.closest('.effect-transition')?.querySelector('.effect-kind')?.textContent;
    viewer.querySelector('h2').textContent = kind ? `${title} · ${kind}` : title;
    viewer.querySelector('.viewer-footer p').textContent = `${position + 1} / ${groups.length}`;
    viewer.querySelector('[data-prev]').disabled = position === 0;
    viewer.querySelector('[data-next]').disabled = position === groups.length - 1;
  };
  document.querySelectorAll('section img').forEach(img => {
    img.classList.add('image-zoom');
    img.tabIndex = 0;
    img.setAttribute('role', 'button');
    img.setAttribute('aria-haspopup', 'dialog');
    img.setAttribute('aria-label', `Enlarge: ${img.alt || 'image'}`);
    img.title = 'Click to enlarge';
    const open = () => {
      opener = img;
      const row = img.closest('.effect-row');
      const sequence = img.closest('.case-attempts, .warmup-stages, .scaling-figs');
      groups = row ? [...row.querySelectorAll('.effect-pair')].map(pair => [...pair.querySelectorAll('img')])
        : sequence ? [...sequence.querySelectorAll('img')].map(item => [item]) : [[img]];
      position = groups.findIndex(group => group.includes(img));
      render();
      viewer.showModal();
      document.body.classList.add('viewer-open');
    };
    img.addEventListener('click', open);
    img.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); } });
  });
  const move = delta => { if (position + delta >= 0 && position + delta < groups.length) { position += delta; render(); } };
  viewer.querySelector('[data-close]').addEventListener('click', () => viewer.close());
  viewer.querySelector('[data-prev]').addEventListener('click', () => move(-1));
  viewer.querySelector('[data-next]').addEventListener('click', () => move(1));
  viewer.addEventListener('keydown', event => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); move(event.key === 'ArrowLeft' ? -1 : 1); }
  });
  viewer.addEventListener('click', event => {
    const box = viewer.getBoundingClientRect();
    if (event.target === viewer && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) viewer.close();
  });
  viewer.addEventListener('close', () => { document.body.classList.remove('viewer-open'); opener?.focus({ preventScroll: true }); });

  const videos = [...document.querySelectorAll('video')];
  videos.forEach(video => {
    video.playsInline = true;
    video.addEventListener('play', () => videos.forEach(other => { if (other !== video) other.pause(); }));
    const showError = () => {
      if (video.nextElementSibling?.classList.contains('video-error')) return;
      const notice = document.createElement('p');
      notice.className = 'video-error';
      notice.setAttribute('role', 'status');
      notice.append('Video could not load. ');
      const link = document.createElement('a');
      link.href = video.querySelector('source')?.src || video.src;
      link.target = '_blank'; link.rel = 'noopener noreferrer'; link.textContent = 'Open original video';
      notice.append(link);
      video.after(notice);
    };
    video.addEventListener('error', showError);
    video.querySelectorAll('source').forEach(source => source.addEventListener('error', showError));
    if (video.error) showError();
  });

})();
