document.addEventListener("DOMContentLoaded", function () {
  const sidebar = document.querySelector(".bd-sidebar-primary");
  if (!sidebar) {
    return;
  }

  // Separate desktop/mobile prefs so they do not overwrite each other.
  const desktopStorageKey = "pst_sidebar_primary_width";
  const mobileStorageKey = "pst_sidebar_mobile_drawer_width";
  const minWidthPx = 200;
  const desktopGutterPx = 24;
  const mobileGutterPx = 16;
  const gutterOverhangPx = 4;
  const desktopQuery = window.matchMedia("(min-width: 960px)");

  const handle = document.createElement("button");
  handle.type = "button";
  handle.className = "bd-sidebar-resize-handle";
  handle.setAttribute("aria-label", "Resize navigation sidebar");
  handle.title = "Drag left or right to resize sidebar";
  document.body.appendChild(handle);

  const primaryToggle = document.querySelector("button.primary-toggle");
  const modal = document.querySelector("#pst-primary-sidebar-modal");

  function isDesktop() {
    return desktopQuery.matches;
  }

  function storageKey() {
    return isDesktop() ? desktopStorageKey : mobileStorageKey;
  }

  function widthVar() {
    return isDesktop()
      ? "--pst-sidebar-primary-width"
      : "--pst-sidebar-mobile-drawer-width";
  }

  function gutterWidthPx() {
    return isDesktop() ? desktopGutterPx : mobileGutterPx;
  }

  function getResizeTarget() {
    if (isDesktop()) {
      return sidebar;
    }
    if (modal && modal.open) {
      return modal;
    }
    return null;
  }

  function maxWidthPx() {
    const ratio = isDesktop() ? 0.5 : 0.9;
    return Math.floor(window.innerWidth * ratio);
  }

  function clampWidth(widthPx) {
    return Math.min(Math.max(widthPx, minWidthPx), maxWidthPx());
  }

  function applyWidth(widthPx, { persist = false } = {}) {
    const target = getResizeTarget();
    if (!target) {
      return;
    }
    const clamped = clampWidth(widthPx);
    document.documentElement.style.setProperty(widthVar(), `${clamped}px`);
    if (persist) {
      localStorage.setItem(storageKey(), String(clamped));
    }
    scheduleSync();
  }

  function syncHandlePosition() {
    const target = getResizeTarget();
    if (!target) {
      handle.hidden = true;
      if (handle.parentElement !== document.body) {
        document.body.appendChild(handle);
      }
      return;
    }

    // Handle must live inside the dialog (top layer) to receive pointer events.
    if (!isDesktop() && modal && modal.open) {
      if (handle.parentElement !== modal) {
        modal.appendChild(handle);
      }
      handle.hidden = false;
      handle.style.position = "absolute";
      handle.style.top = "0";
      handle.style.bottom = "0";
      handle.style.right = `-${gutterOverhangPx}px`;
      handle.style.left = "auto";
      handle.style.width = `${gutterWidthPx()}px`;
      handle.style.height = "auto";
      return;
    }

    if (handle.parentElement !== document.body) {
      document.body.appendChild(handle);
    }

    const rect = target.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      handle.hidden = true;
      return;
    }

    const gutter = gutterWidthPx();
    handle.hidden = false;
    handle.style.position = "fixed";
    handle.style.top = `${rect.top}px`;
    handle.style.left = `${rect.right - gutter + gutterOverhangPx}px`;
    handle.style.right = "auto";
    handle.style.bottom = "auto";
    handle.style.width = `${gutter}px`;
    handle.style.height = `${rect.height}px`;
  }

  function restoreWidth() {
    const saved = localStorage.getItem(storageKey());
    if (!saved) {
      scheduleSync();
      return;
    }
    const widthPx = Number.parseFloat(saved);
    if (!Number.isFinite(widthPx)) {
      scheduleSync();
      return;
    }
    document.documentElement.style.setProperty(
      widthVar(),
      `${clampWidth(widthPx)}px`,
    );
    scheduleSync();
  }

  let syncScheduled = false;
  function scheduleSync() {
    if (syncScheduled) {
      return;
    }
    syncScheduled = true;
    requestAnimationFrame(function () {
      syncScheduled = false;
      syncHeaderOffset();
      syncHandlePosition();
    });
  }

  let dragging = false;
  let startX = 0;
  let startWidth = 0;

  function onPointerMove(event) {
    if (!dragging) {
      return;
    }
    applyWidth(startWidth + (event.clientX - startX));
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

    const target = getResizeTarget();
    if (target) {
      applyWidth(target.getBoundingClientRect().width, { persist: true });
    }
  }

  handle.addEventListener("pointerdown", function (event) {
    const target = getResizeTarget();
    if (!target || event.button !== 0) {
      return;
    }
    event.preventDefault();
    dragging = true;
    startX = event.clientX;
    startWidth = target.getBoundingClientRect().width;
    document.body.classList.add("bd-sidebar-resizing");
    handle.setPointerCapture?.(event.pointerId);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", stopDragging);
    window.addEventListener("pointercancel", stopDragging);
  });

  handle.addEventListener("dblclick", function () {
    localStorage.removeItem(storageKey());
    document.documentElement.style.removeProperty(widthVar());
    scheduleSync();
  });

  window.addEventListener("resize", scheduleSync);
  window.addEventListener("scroll", scheduleSync, { capture: true, passive: true });
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(scheduleSync).observe(sidebar);
    if (modal) {
      new ResizeObserver(scheduleSync).observe(modal);
    }
    const header = document.querySelector(".bd-header");
    if (header) {
      new ResizeObserver(scheduleSync).observe(header);
    }
  }

  restoreWidth();
  scheduleSync();
  desktopQuery.addEventListener("change", function () {
    restoreWidth();
    scheduleSync();
  });

  function closeMobileDrawer() {
    if (isDesktop() || !modal?.open) {
      return;
    }
    if (typeof modal.close === "function") {
      modal.close();
    }
  }

  function syncHeaderOffset() {
    const header = document.querySelector(".bd-header");
    const offset = header ? Math.max(header.getBoundingClientRect().bottom, 0) : 0;
    document.documentElement.style.setProperty(
      "--pst-header-offset",
      `${offset}px`,
    );
    // Lock geometry while open so close doesn't flash full-screen height.
    if (modal && !isDesktop()) {
      modal.style.top = `${offset}px`;
      modal.style.height = `calc(100dvh - ${offset}px)`;
      modal.style.maxHeight = `calc(100dvh - ${offset}px)`;
      modal.style.transform = "none";
      modal.style.transition = "none";
      modal.style.margin = "0";
    }
  }

  modal?.addEventListener("close", scheduleSync);
  modal?.addEventListener("cancel", scheduleSync);
  primaryToggle?.addEventListener("click", function () {
    requestAnimationFrame(scheduleSync);
    setTimeout(scheduleSync, 50);
  });

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "bd-sidebar-mobile-close";
  closeBtn.setAttribute("aria-label", "Close navigation sidebar");
  const closeIcon = document.createElement("i");
  closeIcon.className = "fa-solid fa-xmark";
  closeIcon.setAttribute("aria-hidden", "true");
  closeBtn.appendChild(closeIcon);
  sidebar.prepend(closeBtn);
  closeBtn.addEventListener("click", function (event) {
    event.preventDefault();
    event.stopPropagation();
    closeMobileDrawer();
  });
});
