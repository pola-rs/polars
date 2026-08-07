document.addEventListener("DOMContentLoaded", function () {
  const banner = document.querySelector(".bd-header-announcement");
  if (!banner || banner.dataset.pstAnnouncementUrl) {
    return;
  }

  const storageKey = "pst_announcement_banner_pref";
  const timeoutDays = 14;

  const dismissedStr = JSON.parse(
    localStorage.getItem(storageKey) || "{}",
  )["closed"];
  if (dismissedStr) {
    const daysPassed =
      (new Date() - new Date(dismissedStr)) / (24 * 60 * 60 * 1000);
    if (daysPassed < timeoutDays) {
      return;
    }
  }

  banner.style.display = "flex";

  const closeBtn = document.createElement("a");
  closeBtn.className = "ms-3 my-1 align-baseline";
  closeBtn.style.cursor = "pointer";
  const icon = document.createElement("i");
  icon.className = "fa-solid fa-xmark";
  closeBtn.appendChild(icon);
  closeBtn.addEventListener("click", function () {
    banner.style.display = "none";
    const pref = JSON.parse(localStorage.getItem(storageKey) || "{}");
    pref["closed"] = new Date().toISOString();
    localStorage.setItem(storageKey, JSON.stringify(pref));
  });
  banner.appendChild(closeBtn);
});
