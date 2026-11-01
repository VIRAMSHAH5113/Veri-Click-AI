(() => {
  'use strict';
  if (window.__veriClickInjected) return;
  window.__veriClickInjected = true;

  const API = 'http://127.0.0.1:8000/api/analyze';
  const MAX_CARDS = 12;
  let scanInProgress = false;

  function ensureToolbar() {
    let button = document.getElementById('vericlick-scan-button');
    if (button) return button;
    button = document.createElement('button');
    button.id = 'vericlick-scan-button';
    button.type = 'button';
    button.innerHTML = '<span class="vericlick-scan-glyph">✦</span><span>Scan visible videos</span><span class="vericlick-scan-arrow">↗</span>';
    button.addEventListener('click', scanVisibleVideos);
    document.documentElement.appendChild(button);
    return button;
  }

  function collectCards() {
    const nodes = Array.from(document.querySelectorAll('ytd-rich-item-renderer, ytd-video-renderer, ytd-compact-video-renderer'));
    const seen = new Set();
    const cards = [];
    for (const node of nodes) {
      const titleNode = node.querySelector('#video-title, a#video-title, h3 a[title]');
      const title = (titleNode?.getAttribute('title') || titleNode?.textContent || '').trim().replace(/\s+/g, ' ');
      const imageNode = node.querySelector('ytd-thumbnail img, a#thumbnail img, #thumbnail img');
      const thumbnailUrl = imageNode?.currentSrc || imageNode?.src || '';
      if (title.length < 3 || !/^https:\/\//i.test(thumbnailUrl) || !/ytimg\.com\//i.test(thumbnailUrl)) continue;
      const key = `${title}::${thumbnailUrl}`;
      if (seen.has(key)) continue;
      seen.add(key);
      cards.push({ node, title, thumbnailUrl, imageNode });
      if (cards.length >= MAX_CARDS) break;
    }
    return cards;
  }

  function cardBadgeTarget(card) {
    return card.node.querySelector('a#thumbnail, ytd-thumbnail, #thumbnail') || card.imageNode?.parentElement || card.node;
  }

  function setBadge(card, result) {
    const target = cardBadgeTarget(card);
    if (!target) return;
    target.classList.add('vericlick-position-context');
    let badge = target.querySelector(':scope > .vericlick-risk-badge');
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'vericlick-risk-badge';
      target.appendChild(badge);
    }
    const level = result.risk_score >= 62 ? 'high' : result.risk_score >= 35 ? 'review' : 'low';
    const label = result.risk_score >= 62 ? 'HIGH RISK' : result.risk_score >= 35 ? 'REVIEW' : 'LOW RISK';
    badge.className = `vericlick-risk-badge vericlick-${level}`;
    badge.textContent = `${label} · ${result.risk_score}`;
    badge.title = `${result.label}. Signals: ${(result.reasons || []).slice(0, 3).join('; ')}. Screening aid, not a fact-check verdict.`;
    badge.dataset.riskScore = String(result.risk_score);
  }

  function setButtonState(text, busy = false) {
    const button = ensureToolbar();
    button.disabled = busy;
    button.classList.toggle('vericlick-busy', busy);
    button.innerHTML = `<span class="vericlick-scan-glyph">${busy ? '◌' : '✦'}</span><span>${text}</span>${busy ? '<span class="vericlick-progress"></span>' : '<span class="vericlick-scan-arrow">↗</span>'}`;
  }

  async function scanVisibleVideos() {
    if (scanInProgress) return;
    scanInProgress = true;
    const cards = collectCards();
    if (!cards.length) {
      setButtonState('No video cards found');
      window.setTimeout(() => setButtonState('Scan visible videos'), 2200);
      scanInProgress = false;
      return;
    }
    let completed = 0;
    setButtonState(`Scanning 0 / ${cards.length}`, true);
    for (const card of cards) {
      try {
        const response = await fetch(API, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: card.title, thumbnail_url: card.thumbnailUrl, save_history: false })
        });
        if (response.ok) {
          const result = await response.json();
          setBadge(card, result);
        } else {
          const responseBody = await response.json().catch(() => ({}));
          console.warn('[Veri-Click] Analysis failed:', responseBody.detail || response.status);
        }
      } catch (error) {
        console.warn('[Veri-Click] Could not reach the local API. Start Veri-Click AI first.', error);
        setButtonState('Start local API first');
        window.setTimeout(() => setButtonState('Scan visible videos'), 2800);
        scanInProgress = false;
        return;
      }
      completed++;
      setButtonState(`Scanning ${completed} / ${cards.length}`, true);
      await new Promise(resolve => window.setTimeout(resolve, 100));
    }
    setButtonState(`Scanned ${completed} video${completed === 1 ? '' : 's'}`);
    window.setTimeout(() => setButtonState('Scan visible videos'), 2700);
    scanInProgress = false;
  }

  ensureToolbar();
  // YouTube is a single-page app, so recreate the toolbar if navigation swaps the DOM shell.
  const observer = new MutationObserver(() => {
    if (!document.getElementById('vericlick-scan-button')) ensureToolbar();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
