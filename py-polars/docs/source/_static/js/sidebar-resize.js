document.addEventListener("DOMContentLoaded", function () {
  const sidebar = document.querySelector(".bd-sidebar-primary");
  if (!sidebar) {
    return;
  }

  const storageKey = "pst_sidebar_primary_width";
  const minWidthPx = 200;
  // Left-of-scroll + scrollbar + right-of-scroll (same coverage desktop/mobile).
  const gutterWidthPx = 32;
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
    // Desktop: keep article readable. Mobile drawer: allow a wider panel.
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
    if (isDesktop()) {
      document.documentElement.style.setProperty(
        "--pst-sidebar-primary-width",
        `${clamped}px`,
      );
    } else {
      document.documentElement.style.setProperty(
        "--pst-sidebar-mobile-drawer-width",
        `${clamped}px`,
      );
    }
    if (persist) {
      localStorage.setItem(storageKey, String(clamped));
    }
    syncHandlePosition();
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

    // Dialogs live in the top layer — the handle must be inside the modal on
    // mobile or it cannot receive pointer events over the drawer.
    if (!isDesktop() && modal && modal.open) {
      if (handle.parentElement !== modal) {
        modal.appendChild(handle);
      }
      // Match desktop: no visible grip, cursor-only affordance.
      handle.innerHTML = "";
      delete handle.dataset.mobileGrip;
      handle.hidden = false;
      handle.style.position = "absolute";
      handle.style.top = "0";
      handle.style.bottom = "0";
      handle.style.right = `-${gutterOverhangPx}px`;
      handle.style.left = "auto";
      handle.style.width = `${gutterWidthPx}px`;
      handle.style.height = "auto";
      return;
    }

    if (handle.parentElement !== document.body) {
      document.body.appendChild(handle);
    }
    handle.innerHTML = "";
    delete handle.dataset.mobileGrip;

    const rect = target.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      handle.hidden = true;
      return;
    }

    handle.hidden = false;
    handle.style.position = "fixed";
    handle.style.top = `${rect.top}px`;
    // Overhang a few px past the right edge so "right of scroll" is included.
    handle.style.left = `${rect.right - gutterWidthPx + gutterOverhangPx}px`;
    handle.style.right = "auto";
    handle.style.bottom = "auto";
    handle.style.width = `${gutterWidthPx}px`;
    handle.style.height = `${rect.height}px`;
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
    // Apply even before open on mobile so the next open uses the saved width.
    const clamped = clampWidth(widthPx);
    if (isDesktop()) {
      document.documentElement.style.setProperty(
        "--pst-sidebar-primary-width",
        `${clamped}px`,
      );
    } else {
      document.documentElement.style.setProperty(
        "--pst-sidebar-mobile-drawer-width",
        `${clamped}px`,
      );
    }
    syncHandlePosition();
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
    localStorage.removeItem(storageKey);
    document.documentElement.style.removeProperty("--pst-sidebar-primary-width");
    document.documentElement.style.removeProperty(
      "--pst-sidebar-mobile-drawer-width",
    );
    syncHandlePosition();
  });

  window.addEventListener("resize", syncHandlePosition);
  window.addEventListener("scroll", syncHandlePosition, true);
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(syncHandlePosition).observe(sidebar);
    if (modal) {
      new ResizeObserver(syncHandlePosition).observe(modal);
    }
  }

  restoreWidth();
  syncHandlePosition();
  desktopQuery.addEventListener("change", function () {
    restoreWidth();
    syncHandlePosition();
  });

  function closeMobileDrawer() {
    if (isDesktop()) {
      return;
    }
    if (modal && modal.open && typeof modal.close === "function") {
      modal.close();
      return;
    }
    primaryToggle?.click();
  }

  function afterDrawerToggle() {
    requestAnimationFrame(function () {
      syncHeaderOffset();
      syncHandlePosition();
    });
    setTimeout(function () {
      syncHeaderOffset();
      syncHandlePosition();
    }, 50);
  }

  function syncHeaderOffset() {
    const header = document.querySelector(".bd-header");
    const offset = header ? Math.max(header.getBoundingClientRect().bottom, 0) : 0;
    document.documentElement.style.setProperty(
      "--pst-header-offset",
      `${offset}px`,
    );
    // Inline-lock modal geometry so close can't flash full-screen height when
    // the [open] attribute drops before the dialog is fully hidden.
    if (modal && !isDesktop()) {
      modal.style.top = `${offset}px`;
      modal.style.height = `calc(100dvh - ${offset}px)`;
      modal.style.maxHeight = `calc(100dvh - ${offset}px)`;
      modal.style.transform = "none";
      modal.style.transition = "none";
      modal.style.margin = "0";
    }
  }
  window.addEventListener("resize", syncHeaderOffset);
  window.addEventListener("scroll", syncHeaderOffset, true);
  if (typeof ResizeObserver !== "undefined") {
    const header = document.querySelector(".bd-header");
    if (header) {
      new ResizeObserver(syncHeaderOffset).observe(header);
    }
  }
  syncHeaderOffset();
  desktopQuery.addEventListener("change", syncHeaderOffset);
  modal?.addEventListener("close", function () {
    syncHeaderOffset();
    syncHandlePosition();
  });
  primaryToggle?.addEventListener("click", afterDrawerToggle);

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "bd-sidebar-mobile-close";
  closeBtn.setAttribute("aria-label", "Close navigation sidebar");
  closeBtn.innerHTML = '<i class="fa-solid fa-xmark" aria-hidden="true"></i>';
  sidebar.prepend(closeBtn);
  closeBtn.addEventListener("click", function (event) {
    event.preventDefault();
    event.stopPropagation();
    closeMobileDrawer();
    afterDrawerToggle();
  });

  // Close when clicking outside the drawer (dimmed backdrop, header, page).
  document.addEventListener(
    "pointerdown",
    function (event) {
      if (isDesktop() || !modal?.open || dragging) {
        return;
      }
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      // Clicks on drawer content / resize handle should not close.
      if (modal.contains(target) && target !== modal) {
        return;
      }
      // Let the hamburger handle its own toggle.
      if (primaryToggle && primaryToggle.contains(target)) {
        return;
      }
      closeMobileDrawer();
      afterDrawerToggle();
    },
    true,
  );
});
