import logging
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)


def load_css() -> None:
    """Inject stylesheets from styles/ partials, concatenated in order."""
    styles_dir = Path(__file__).parent.parent / "styles"
    partials = [
        "_tokens.css",
        "_streamlit-chrome.css",
        "_base.css",
        "_sidebar.css",
        "_welcome.css",
        "_messages.css",
        "_chat-input.css",
        "_loading.css",
        "_responsive.css",
    ]
    chunks: list[str] = []
    for name in partials:
        path = styles_dir / name
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("CSS partial not found: %s — skipping", path)
    if not chunks:
        logger.warning("No CSS partials found in %s — running unstyled", styles_dir)
        return
    st.markdown(f"<style>{''.join(chunks)}</style>", unsafe_allow_html=True)


def install_client_behaviors() -> None:
    """Install all client-side JS behaviors via a single zero-height iframe.

    A version guard on the parent document ensures this only installs once
    per page load, even though Streamlit re-runs this function on every
    interaction.

    Behaviors installed:
      - Snap-to-new-question: when a new .user-row appears, scroll it into
        view with a rAF loop + ResizeObserver retry chain.
      - Jump-to-latest button: floats above the input, fades in when scrolled
        up, snaps back to bottom on click.
      - Bottom-pin while streaming: follow new tokens unless the user scrolled
        up (hold window suppresses this during a snap).
      - Copy button: copies assistant answer text to clipboard.
      - Sources toggle: open/close the citation disclosure panel.
      - Citation chip click: opens panel and pulses the matching source row.
      - Auto-focus: focuses the chat input on first load.
    """
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const win = window.parent;
          const VERSION = 8;
          if (doc.__ragClientVersion === VERSION) return;
          doc.__ragClientVersion = VERSION;
          try { console.log('[RAG] client behaviors installing v' + VERSION); } catch (e) {}

          const NEAR = 140;
          const TOP_PAD = 28;
          const HOLD_MS = 3000;
          const SNAP_FRAMES = 180;
          const WARMUP_MS = 1500;

          /* ---- scroller detection ---- */
          function findScrollerFrom(node) {
            let el = node && node.parentElement;
            while (el && el !== doc.documentElement) {
              const cs = win.getComputedStyle(el);
              const ov = cs.overflowY;
              if ((ov === 'auto' || ov === 'scroll' || ov === 'overlay')
                  && el.scrollHeight > el.clientHeight + 1) {
                return el;
              }
              el = el.parentElement;
            }
            return doc.scrollingElement || doc.documentElement;
          }

          function candidateScrollers() {
            const out = [];
            const sels = [
              '[data-testid="stMain"]',
              'section.main',
              '[data-testid="stAppViewContainer"]',
              'main',
              '[data-testid="ScrollToBottomContainer"]',
              '.main',
            ];
            sels.forEach(function (s) {
              doc.querySelectorAll(s).forEach(function (el) {
                if (out.indexOf(el) === -1) out.push(el);
              });
            });
            if (doc.scrollingElement && out.indexOf(doc.scrollingElement) === -1)
              out.push(doc.scrollingElement);
            if (doc.documentElement && out.indexOf(doc.documentElement) === -1)
              out.push(doc.documentElement);
            return out;
          }

          function atBottomAny() {
            return candidateScrollers().some(function (sc) {
              try { return sc.scrollHeight - sc.scrollTop - sc.clientHeight <= NEAR; }
              catch (e) { return false; }
            });
          }

          function toBottomAll() {
            const lastMsg = doc.querySelector('.user-row:last-of-type')
                         || doc.querySelector('[data-testid="stChatMessage"]:last-of-type')
                         || doc.body.lastElementChild;
            const list = candidateScrollers();
            const primary = findScrollerFrom(lastMsg);
            if (primary && list.indexOf(primary) === -1) list.unshift(primary);
            list.forEach(function (sc) {
              try {
                const max = Math.max(0, sc.scrollHeight - sc.clientHeight);
                if (sc.scrollTop < max - 1) sc.scrollTop = max;
              } catch (e) {}
            });
            try { win.scrollTo(0, 99999); } catch (e) {}
          }

          function snapToRow(row) {
            if (!row || !doc.body.contains(row)) return;
            const rRect = row.getBoundingClientRect();
            const primary = findScrollerFrom(row);
            const list = candidateScrollers();
            if (primary && list.indexOf(primary) === -1) list.unshift(primary);
            list.forEach(function (sc) {
              try {
                const isDoc = (sc === doc.scrollingElement
                            || sc === doc.documentElement
                            || sc === doc.body);
                const scTop = isDoc ? 0 : sc.getBoundingClientRect().top;
                const delta = rRect.top - scTop - TOP_PAD;
                if (Math.abs(delta) < 0.5) return;
                const max = Math.max(0, sc.scrollHeight - sc.clientHeight);
                const next = Math.max(0, Math.min(max, sc.scrollTop + delta));
                if (Math.abs(next - sc.scrollTop) > 0.5) sc.scrollTop = next;
              } catch (e) {}
            });
            try {
              const desired = win.scrollY + (rRect.top - TOP_PAD);
              if (Math.abs(desired - win.scrollY) > 0.5)
                win.scrollTo({ top: Math.max(0, desired), behavior: 'auto' });
            } catch (e) {}
          }

          function holdActive() {
            return win.__ragHoldUntil && Date.now() < win.__ragHoldUntil;
          }

          /* ---- jump-to-latest button ---- */
          const jump = doc.createElement('button');
          jump.type = 'button';
          jump.className = 'jump-latest';
          jump.setAttribute('aria-label', 'Jump to latest');
          jump.innerHTML =
              '<svg viewBox="0 0 24 24" width="16" height="16" fill="none"'
            + ' stroke="currentColor" stroke-width="2" stroke-linecap="round"'
            + ' stroke-linejoin="round"><path d="M12 5v14"></path>'
            + '<path d="M19 12l-7 7-7-7"></path></svg>';
          doc.body.appendChild(jump);

          function placeJump() {
            const pill = doc.querySelector('[data-testid="stChatInput"]')
                      || doc.querySelector('[data-testid="stBottom"]');
            if (pill) {
              const r = pill.getBoundingClientRect();
              jump.style.left = (r.left + r.width / 2) + 'px';
              jump.style.bottom = (win.innerHeight - r.top + 10) + 'px';
            } else {
              jump.style.left = '50%';
              jump.style.bottom = '110px';
            }
          }
          jump.addEventListener('click', function () {
            pinned = true;
            win.__ragHoldUntil = 0;
            toBottomAll();
            updateJump();
          });

          let pinned = true;
          function updateJump() {
            placeJump();
            jump.classList.toggle('is-visible', !pinned);
          }

          /* ---- scroll + gesture listeners ---- */
          let sQueued = false;
          function onScroll() {
            if (sQueued) return;
            sQueued = true;
            win.requestAnimationFrame(function () {
              sQueued = false;
              pinned = atBottomAny();
              updateJump();
            });
          }
          candidateScrollers().forEach(function (t) {
            try { t.addEventListener('scroll', onScroll, { passive: true }); } catch (e) {}
          });
          win.addEventListener('scroll', onScroll, { passive: true });
          win.addEventListener('resize', placeJump);

          function releaseHold() { win.__ragHoldUntil = 0; }
          ['wheel', 'touchstart', 'touchmove'].forEach(function (ev) {
            win.addEventListener(ev, releaseHold, { passive: true });
          });
          win.addEventListener('keydown', function (e) {
            const t = e.target;
            if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' || t.isContentEditable)) return;
            if (e.key === 'ArrowUp' || e.key === 'ArrowDown'
             || e.key === 'PageUp' || e.key === 'PageDown'
             || e.key === 'Home' || e.key === 'End' || e.key === ' ') {
              releaseHold();
            }
          }, true);

          /* ---- snap driver ---- */
          let activeSnapRow = null;
          let activeSnapRO = null;

          function startSnap(row) {
            if (!row) return;
            activeSnapRow = row;
            win.__ragHoldUntil = Date.now() + HOLD_MS;
            pinned = false;
            updateJump();
            snapToRow(row);
            let frames = 0;
            (function loop() {
              if (activeSnapRow !== row) return;
              frames++;
              snapToRow(row);
              if (frames < SNAP_FRAMES && Date.now() < win.__ragHoldUntil)
                win.requestAnimationFrame(loop);
            })();
            [80, 200, 400, 800, 1400, 2200, 2800].forEach(function (ms) {
              setTimeout(function () {
                if (activeSnapRow === row) snapToRow(row);
              }, ms);
            });
            if (win.ResizeObserver) {
              if (activeSnapRO) { try { activeSnapRO.disconnect(); } catch (e) {} }
              activeSnapRO = new win.ResizeObserver(function () {
                if (activeSnapRow === row && Date.now() < win.__ragHoldUntil)
                  snapToRow(row);
              });
              try { activeSnapRO.observe(row); } catch (e) {}
              try { activeSnapRO.observe(doc.body); } catch (e) {}
              setTimeout(function () {
                try { activeSnapRO.disconnect(); } catch (e) {}
                if (activeSnapRow === row) activeSnapRow = null;
              }, HOLD_MS + 500);
            } else {
              setTimeout(function () {
                if (activeSnapRow === row) activeSnapRow = null;
              }, HOLD_MS + 500);
            }
          }

          /* ---- detection via count growth ---- */
          let armed = false;
          let lastRowCount = 0;
          function readCount() { return doc.querySelectorAll('.user-row').length; }
          setTimeout(function () {
            armed = true;
            lastRowCount = readCount();
          }, WARMUP_MS);

          let mQueued = false;
          const obs = new MutationObserver(function () {
            if (mQueued) return;
            mQueued = true;
            win.requestAnimationFrame(function () {
              mQueued = false;
              if (armed) {
                const c = readCount();
                if (c > lastRowCount) {
                  const rows = doc.querySelectorAll('.user-row');
                  const newest = rows[rows.length - 1];
                  lastRowCount = c;
                  if (newest && newest !== activeSnapRow) startSnap(newest);
                } else if (c < lastRowCount) {
                  lastRowCount = c;
                }
              }
              if (holdActive()) { updateJump(); return; }
              if (pinned) toBottomAll();
              updateJump();
            });
          });
          obs.observe(doc.body, { childList: true, subtree: true, characterData: true });

          win.__ragSnapToLatest = function () {
            const rows = doc.querySelectorAll('.user-row');
            if (!rows.length) return;
            const newest = rows[rows.length - 1];
            if (newest && newest !== activeSnapRow) startSnap(newest);
          };

          /* ---- copy button ---- */
          doc.addEventListener('click', function (e) {
            const btn = e.target.closest && e.target.closest('.copy-btn');
            if (!btn) return;
            const content = btn.closest('[data-testid="stChatMessageContent"]')
                         || btn.closest('[data-testid="stChatMessage"]');
            if (!content) return;
            const clone = content.cloneNode(true);
            clone.querySelectorAll('.msg-actions').forEach(function (n) { n.remove(); });
            const text = (clone.innerText || '').trim();
            function flash() {
              btn.classList.add('copied');
              const label = btn.querySelector('.copy-label');
              const prev = label ? label.textContent : '';
              if (label) label.textContent = 'Copied';
              setTimeout(function () {
                btn.classList.remove('copied');
                if (label) label.textContent = prev || 'Copy';
              }, 1400);
            }
            if (win.navigator.clipboard && win.navigator.clipboard.writeText) {
              win.navigator.clipboard.writeText(text).then(flash).catch(function () {});
            } else {
              const ta = doc.createElement('textarea');
              ta.value = text;
              ta.style.position = 'fixed';
              ta.style.opacity = '0';
              doc.body.appendChild(ta);
              ta.select();
              try { doc.execCommand('copy'); flash(); } catch (err) {}
              doc.body.removeChild(ta);
            }
          }, true);

          /* ---- sources toggle ---- */
          doc.addEventListener('click', function (e) {
            const t = e.target.closest && e.target.closest('.sources-toggle');
            if (!t) return;
            const box = t.closest('.sources');
            if (!box) return;
            const open = box.classList.toggle('open');
            t.setAttribute('aria-expanded', open ? 'true' : 'false');
          }, true);

          /* ---- citation chip click ---- */
          doc.addEventListener('click', function (e) {
            const chip = e.target.closest && e.target.closest('.cite');
            if (!chip) return;
            const docname = chip.dataset.doc;
            if (!docname) return;
            const root = chip.closest('[data-testid="stChatMessageContent"]')
                      || chip.closest('[data-testid="stChatMessage"]')
                      || doc;
            const box = root.querySelector('.sources');
            if (!box) return;
            if (!box.classList.contains('open')) {
              box.classList.add('open');
              const tog = box.querySelector('.sources-toggle');
              if (tog) tog.setAttribute('aria-expanded', 'true');
            }
            const items = box.querySelectorAll('.source-item');
            items.forEach(function (it) { it.classList.remove('is-pulsing'); });
            setTimeout(function () {
              let matched = null;
              items.forEach(function (it) {
                if (!matched && it.dataset.doc === docname) matched = it;
              });
              if (matched) {
                matched.classList.add('is-pulsing');
                setTimeout(function () { matched.classList.remove('is-pulsing'); }, 1800);
              }
            }, 320);
          }, true);
          doc.addEventListener('keydown', function (e) {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            const chip = e.target.closest && e.target.closest('.cite');
            if (!chip) return;
            e.preventDefault();
            chip.click();
          });

          /* ---- auto-focus chat input ---- */
          (function focusOnce(n) {
            const ta = doc.querySelector('[data-testid="stChatInput"] textarea');
            if (ta) { ta.focus(); }
            else if (n > 0) { setTimeout(function () { focusOnce(n - 1); }, 150); }
          })(20);

          placeJump();
          if (!holdActive()) toBottomAll();
          try { console.log('[RAG] client behaviors ready'); } catch (e) {}
        })();
        </script>
        """,
        height=0,
    )
