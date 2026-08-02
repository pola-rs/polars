document.addEventListener("DOMContentLoaded", function () {
  const sidebar = document.querySelector(".bd-sidebar-primary");
  if (!sidebar) {
    return;
  }

  const storageKey = "pst_sidebar_primary_width";
  const minWidthPx = 200;
  const maxWidthRatio = 0.5;
  const gutterWidthPx = 16;
  const desktopQuery = window.matchMedia("(min-width: 960px)");

  // Fixed overlay pinned to the scrollbar gutter (native scrollbars sit above
  // in-sidebar elements, so this must live outside the sidebar).
  const handle = document.createElement("button");
  handle.type = "button";
  handle.className = "bd-sidebar-resize-handle";
  handle.setAttribute("aria-label", "Resize navigation sidebar");
  handle.title = "Drag left or right to resize sidebar";
  document.body.appendChild(handle);

  function isDesktop() {
    return desktopQuery.matches;
  }

  function syncHandlePosition() {
    if (!isDesktop()) {
      handle.hidden = true;
      return;
    }

    const rect = sidebar.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      handle.hidden = true;
      return;
    }

    // Cover the scrollbar track and a couple of pixels past its right edge.
    handle.hidden = false;
    handle.style.top = `${rect.top}px`;
    handle.style.left = `${rect.right - gutterWidthPx}px`;
    handle.style.width = `${gutterWidthPx + 2}px`;
    handle.style.height = `${rect.height}px`;
  }

  function clampWidth(widthPx) {
    const maxWidthPx = Math.floor(window.innerWidth * maxWidthRatio);
    return Math.min(Math.max(widthPx, minWidthPx), maxWidthPx);
  }

  function applyWidth(widthPx, { persist = false } = {}) {
    if (!isDesktop()) {
      return;
    }
    const clamped = clampWidth(widthPx);
    document.documentElement.style.setProperty(
      "--pst-sidebar-primary-width",
      `${clamped}px`,
    );
    if (persist) {
      localStorage.setItem(storageKey, String(clamped));
    }
    syncHandlePosition();
  }

  function restoreWidth() {
    const saved = localStorage.getItem(storageKey);
    if (!saved) {
      syncHandlePosition();
      return;
    }
    const widthPx = Number.parseFloat(saved);
    if (!Number.isFinite(widthPx)) {
      syncHandlePosition();
      return;
    }
    applyWidth(widthPx);
  }

  let dragging = false;
  let startX = 0;
  let startWidth = 0;

  function onPointerMove(event) {
    if (!dragging) {
      return;
    }
    const delta = event.clientX - startX;
    applyWidth(startWidth + delta);
  }

  function stopDragging() {
    if (!dragging) {
      return;
    }
    dragging = false;
    document.body.classList.remove("bd-sidebar-resizing");
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", stopDragging);
    window.removeEventListener("pointercancel", stopDragging);

    const widthPx = sidebar.getBoundingClientRect().width;
    applyWidth(widthPx, { persist: true });
  }

  handle.addEventListener("pointerdown", function (event) {
    if (!isDesktop() || event.button !== 0) {
      return;
    }
    event.preventDefault();
    dragging = true;
    startX = event.clientX;
    startWidth = sidebar.getBoundingClientRect().width;
    document.body.classList.add("bd-sidebar-resizing");
    handle.setPointerCapture?.(event.pointerId);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", stopDragging);
    window.addEventListener("pointercancel", stopDragging);
  });

  // Double-click resets to the default fluid width.
  handle.addEventListener("dblclick", function () {
    localStorage.removeItem(storageKey);
    document.documentElement.style.removeProperty("--pst-sidebar-primary-width");
    syncHandlePosition();
  });

  window.addEventListener("resize", syncHandlePosition);
  window.addEventListener("scroll", syncHandlePosition, true);
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(syncHandlePosition).observe(sidebar);
  }

  restoreWidth();
  syncHandlePosition();
  desktopQuery.addEventListener("change", function () {
    if (isDesktop()) {
      restoreWidth();
    } else {
      document.documentElement.style.removeProperty("--pst-sidebar-primary-width");
      syncHandlePosition();
    }
  });
});
