const revealTargets = document.querySelectorAll(
  ".section-heading, .hero-copy, .hero-panel, .profile-card, .anniversary-intro, .anniversary-card, .station-card, .station-banner, .preview-window, .preview-poster, .video-hero-copy, .video-hero-panel, .video-poster, .video-card, .video-link-card, .video-note-card, .community-intro, .community-card, .link-card, .feature-card, .media-card, .note-card, .timeline-item, .quote-panel, .site-footer"
);

revealTargets.forEach((element) => {
  element.setAttribute("data-reveal", "");
});

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },
  {
    threshold: 0.16,
    rootMargin: "0px 0px -40px 0px",
  }
);

revealTargets.forEach((element) => observer.observe(element));

const anniversaryCards = document.querySelectorAll(".anniversary-card");

const startOfDay = (date) =>
  new Date(date.getFullYear(), date.getMonth(), date.getDate());

anniversaryCards.forEach((card) => {
  const month = Number(card.dataset.month);
  const day = Number(card.dataset.day);
  const startYear = Number(card.dataset.startYear);
  const ddayNode = card.querySelector(".anniversary-dday");
  const cycleNode = card.querySelector(".anniversary-cycle");

  if (!month || !day || !startYear || !ddayNode || !cycleNode) {
    return;
  }

  const today = startOfDay(new Date());
  let target = new Date(today.getFullYear(), month - 1, day);

  if (target < today) {
    target = new Date(today.getFullYear() + 1, month - 1, day);
  }

  const diffDays = Math.round((target - today) / 86400000);
  const anniversaryYear = target.getFullYear() - startYear;

  if (diffDays === 0) {
    ddayNode.textContent = "D-Day";
  } else {
    ddayNode.textContent = `D-${diffDays}`;
  }

  cycleNode.textContent = `${anniversaryYear}주년 · ${target.getFullYear()}.${String(
    month
  ).padStart(2, "0")}.${String(day).padStart(2, "0")}`;
});
