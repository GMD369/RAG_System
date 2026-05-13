// ── Utilities ────────────────────────────────────────────────

function showEl(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = '';
}

function hideEl(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = 'none';
}

function escHtml(str) {
  const d = document.createElement('div');
  d.textContent = String(str);
  return d.innerHTML;
}

async function postJSON(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Pipeline flow animation ──────────────────────────────────

function resetFlow(flowId, steps) {
  for (let i = 1; i <= steps; i++) {
    const node = document.getElementById(`pn-${i}`);
    const con  = document.getElementById(`con-${i}`);
    if (node) node.classList.remove('active', 'complete', 'error');
    if (con)  con.classList.remove('active');
  }
}

/**
 * Animates pipeline nodes one-by-one.
 * Returns an interval ID so the caller can clearInterval when done.
 */
function animateFlow(flowId, steps, stepMs = 800) {
  let current = 1;

  const tick = () => {
    if (current > steps) return;

    // Mark previous node complete
    if (current > 1) {
      const prev = document.getElementById(`pn-${current - 1}`);
      if (prev) { prev.classList.remove('active'); prev.classList.add('complete'); }
    }

    // Activate current node
    const node = document.getElementById(`pn-${current}`);
    if (node) node.classList.add('active');

    // Activate connector before current node
    const con = document.getElementById(`con-${current - 1}`);
    if (con) con.classList.add('active');

    current++;
  };

  tick(); // immediately activate first node
  return setInterval(tick, stepMs);
}

function completeFlow(flowId, steps) {
  for (let i = 1; i <= steps; i++) {
    const node = document.getElementById(`pn-${i}`);
    const con  = document.getElementById(`con-${i}`);
    if (node) { node.classList.remove('active'); node.classList.add('complete'); }
    if (con)  con.classList.add('active');
  }
}

function errorFlow(flowId) {
  // Mark any still-active node as error
  document.querySelectorAll(`#${flowId} .pnode.active`).forEach(n => {
    n.classList.remove('active');
    n.classList.add('error');
  });
}
