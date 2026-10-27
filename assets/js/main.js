(function () {
  'use strict';

  // Gallery data: [width, height] per image, files are numbered 00, 01, ...
  var DIMS = {
    t2i: [[1024,1024],[1024,1024],[1024,1024],[1024,1024],[720,1280],[720,1280],[720,1280],[720,1280],[1280,720],[1280,720],[1280,549],[1280,549],[1280,549],[1280,549]],
    edit: [[761,1024],[808,1024],[1024,732],[838,838],[1024,726],[828,828],[721,721],[536,721],[721,721],[659,659],[659,659],[659,659],[1003,1003],[1003,1003],[1003,1003]]
  };
  function pad(i) { return (i < 10 ? '0' : '') + i; }
  function items(kind) {
    return DIMS[kind].map(function (d, i) {
      return { src: 'assets/gallery/' + kind + '/' + pad(i) + '.webp', w: d[0], h: d[1] };
    });
  }
  var T2I = items('t2i');
  var EDIT = DIMS.edit.map(function (d, i) {
    var base = 'assets/gallery/edit/' + pad(i);
    return { inp: base + '_in.webp', out: base + '_out.webp', w: d[0], h: d[1] };
  });

  function el(tag, attrs) {
    var e = document.createElement(tag);
    for (var k in attrs) { if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, attrs[k]); }
    return e;
  }

  // ---------- marquee ----------
  var track = document.getElementById('marquee-track');
  if (track) {
    var picks = T2I;

    // Two identical sheets make the full-width strip loop seamlessly.
    for (var copy = 0; copy < 2; copy++) {
      var sheet = el('div', { 'class': 'marquee-sheet' });
      picks.forEach(function (it) {
        var img = el('img', { src: it.src, alt: '', width: it.w, height: it.h, loading: copy ? 'lazy' : 'eager', decoding: 'async' });
        img.style.setProperty('height', '180px', 'important');
        img.style.setProperty('width', 'auto', 'important');
        sheet.appendChild(img);
      });
      track.appendChild(sheet);
    }
  }

  // ---------- lightbox ----------
  var lb = document.getElementById('lightbox');
  var lbImg = document.getElementById('lb-img');
  var lbList = [], lbIdx = 0, lbReturn = null;
  function lbShow(i) {
    lbIdx = (i + lbList.length) % lbList.length;
    lbImg.src = lbList[lbIdx].src;
    lbImg.alt = lbList[lbIdx].alt;
  }
  function lbOpen(list, i, opener) {
    lbList = list; lbReturn = opener;
    lbShow(i);
    lb.classList.add('open');
    document.body.style.overflow = 'hidden';
    lb.querySelector('.lb-close').focus();
  }
  function lbClose() {
    lb.classList.remove('open');
    document.body.style.overflow = '';
    lbImg.removeAttribute('src');
    if (lbReturn) lbReturn.focus();
  }
  lb.querySelector('.lb-close').addEventListener('click', lbClose);
  lb.querySelector('.lb-prev').addEventListener('click', function () { lbShow(lbIdx - 1); });
  lb.querySelector('.lb-next').addEventListener('click', function () { lbShow(lbIdx + 1); });
  lb.addEventListener('click', function (e) { if (e.target === lb) lbClose(); });
  document.addEventListener('keydown', function (e) {
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape') lbClose();
    else if (e.key === 'ArrowLeft') lbShow(lbIdx - 1);
    else if (e.key === 'ArrowRight') lbShow(lbIdx + 1);
    else if (e.key === 'Tab') {
      // keep focus inside the dialog
      var f = lb.querySelectorAll('button');
      var first = f[0], last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  });

  // ---------- masonry grids ----------
  function renderMasonry(id, list, label) {
    var root = document.getElementById(id);
    if (!root) return;
    var lbItems = list.map(function (it, i) { return { src: it.src, alt: label + ' ' + (i + 1) }; });
    list.forEach(function (it, i) {
      var b = el('button', { type: 'button', 'class': 'tile', 'aria-label': 'Enlarge ' + lbItems[i].alt });
      b.appendChild(el('img', { src: it.src, alt: lbItems[i].alt, width: it.w, height: it.h, loading: 'lazy', decoding: 'async' }));
      b.addEventListener('click', function () { lbOpen(lbItems, i, b); });
      root.appendChild(b);
    });
  }
  renderMasonry('grid-t2i', T2I, 'Text-to-image sample');

  // ---------- edit compare sliders ----------
  var editRoot = document.getElementById('grid-edit');
  if (editRoot) {
    EDIT.forEach(function (it, i) {
      var box = el('div', { 'class': 'compare' });
      box.style.aspectRatio = it.w + ' / ' + it.h;
      box.appendChild(el('img', { src: it.inp, alt: 'Editing example ' + (i + 1) + ', input', width: it.w, height: it.h, loading: 'lazy', decoding: 'async' }));
      var after = el('img', { 'class': 'after', src: it.out, alt: 'Editing example ' + (i + 1) + ', HiUni output', width: it.w, height: it.h, loading: 'lazy', decoding: 'async' });
      box.appendChild(after);
      box.appendChild(el('div', { 'class': 'handle', 'aria-hidden': 'true' }));
      var tIn = el('span', { 'class': 'tag in', 'data-i18n': 'g.in' }); tIn.textContent = 'Input';
      var tOut = el('span', { 'class': 'tag out', 'data-i18n': 'g.out' }); tOut.textContent = 'Output';
      box.appendChild(tIn); box.appendChild(tOut);
      var range = el('input', { type: 'range', min: '0', max: '100', value: '50', 'aria-label': 'Compare input and output, example ' + (i + 1) });
      range.addEventListener('input', function () { box.style.setProperty('--pos', range.value + '%'); });
      box.appendChild(range);
      editRoot.appendChild(box);
    });
  }
  // ---------- tabs ----------
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tabs [role="tab"]'));
  function selectTab(tab, focus) {
    tabs.forEach(function (t) {
      var on = t === tab;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
    if (focus) tab.focus();
  }
  tabs.forEach(function (t, i) {
    t.addEventListener('click', function () { selectTab(t, false); });
    t.addEventListener('keydown', function (e) {
      var n = null;
      if (e.key === 'ArrowRight') n = tabs[(i + 1) % tabs.length];
      else if (e.key === 'ArrowLeft') n = tabs[(i - 1 + tabs.length) % tabs.length];
      else if (e.key === 'Home') n = tabs[0];
      else if (e.key === 'End') n = tabs[tabs.length - 1];
      if (n) { e.preventDefault(); selectTab(n, true); }
    });
  });

  // ---------- copy buttons ----------
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      var ta = el('textarea', { readonly: '' });
      ta.value = text;
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy') ? resolve() : reject(); } catch (err) { reject(err); }
      document.body.removeChild(ta);
    });
  }
  document.querySelectorAll('.copy-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var code = btn.parentNode.querySelector('pre');
      var label = btn.textContent;
      copyText(code.innerText.replace(/\n$/, '')).then(function () {
        btn.textContent = t('q.copied', 'Copied');
      }, function () {
        btn.textContent = t('q.copyfail', 'Failed');
      }).then(function () {
        setTimeout(function () { btn.textContent = label; }, 1400);
      });
    });
  });

  // ---------- reveal on scroll ----------
  var reveals = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    reveals.forEach(function (r) { io.observe(r); });
  } else {
    reveals.forEach(function (r) { r.classList.add('in'); });
  }

  // ---------- i18n ----------
  // English text is read from the page itself; ZH holds the translations.
  var ZH = window.HIUNI_ZH || {};
  var LANG_KEY = 'hiuni_site_lang';
  var THEME_KEY = 'hiuni_site_theme';
  var lang = 'en';
  var EN = {};
  function t(key, fallback) {
    if (lang === 'zh' && ZH[key]) return ZH[key];
    return EN[key] || fallback;
  }
  function nodes() { return document.querySelectorAll('[data-i18n],[data-i18n-html]'); }
  function capture() {
    nodes().forEach(function (n) {
      var k = n.getAttribute('data-i18n-html') || n.getAttribute('data-i18n');
      if (!(k in EN)) EN[k] = n.hasAttribute('data-i18n-html') ? n.innerHTML : n.textContent;
    });
    EN['q.copied'] = 'Copied'; EN['q.copyfail'] = 'Failed';
  }
  function setSiteLang(next) {
    lang = next === 'zh' ? 'zh' : 'en';
    var dict = lang === 'zh' ? ZH : EN;
    nodes().forEach(function (n) {
      var html = n.hasAttribute('data-i18n-html');
      var k = html ? n.getAttribute('data-i18n-html') : n.getAttribute('data-i18n');
      var v = dict[k] != null ? dict[k] : EN[k];
      if (v == null) return;
      if (html) n.innerHTML = v; else n.textContent = v;
    });
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    var btn = document.getElementById('site-lang-toggle');
    if (btn) {
      btn.textContent = lang === 'zh' ? 'EN' : '中文';
      btn.setAttribute('aria-label', lang === 'zh' ? 'Switch to English' : '切换到中文');
    }
    try { localStorage.setItem(LANG_KEY, lang); } catch (err) { /* storage disabled */ }
  }
  function setSiteTheme(next) {
    var theme = next === 'dark' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.style.colorScheme = theme;
    var btn = document.getElementById('theme-toggle');
    if (btn) {
      var dark = theme === 'dark';
      btn.querySelector('.theme-icon').textContent = dark ? '☀' : '☾';
      btn.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
      btn.setAttribute('title', dark ? 'Switch to light mode' : 'Switch to dark mode');
      btn.setAttribute('aria-pressed', dark ? 'true' : 'false');
    }
    try { localStorage.setItem(THEME_KEY, theme); } catch (err) { /* storage disabled */ }
  }
  capture();
  var saved = null;
  try { saved = localStorage.getItem(LANG_KEY); } catch (err) { /* storage disabled */ }
  if (saved === 'zh') setSiteLang('zh');
  var toggle = document.getElementById('site-lang-toggle');
  if (toggle) toggle.addEventListener('click', function () { setSiteLang(lang === 'zh' ? 'en' : 'zh'); });
  var savedTheme = null;
  try { savedTheme = localStorage.getItem(THEME_KEY); } catch (err) { /* storage disabled */ }
  setSiteTheme(savedTheme === 'dark' ? 'dark' : 'light');
  var themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) themeToggle.addEventListener('click', function () {
    setSiteTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });
})();
