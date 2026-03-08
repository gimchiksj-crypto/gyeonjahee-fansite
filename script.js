const revealTargets = document.querySelectorAll(
  ".section-heading, .hero-copy, .hero-panel, .hero-route-card, .priority-card, .hub-card, .goods-intro, .goods-card, .profile-card, .anniversary-intro, .anniversary-card, .station-card, .preview-window, .preview-poster, .video-hero-copy, .video-hero-panel, .video-poster, .video-link-card, .community-intro, .community-card, .link-card, .feature-card, .media-card, .note-card, .timeline-item, .quote-panel, .cta-card, .site-footer"
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
const countdownChips = document.querySelectorAll(".countdown-chip");
const liveStatusNode = document.querySelector("[data-live-status]");
const liveToggleButton = document.querySelector("[data-live-toggle]");
const birthdaySummaryNode = document.querySelector("[data-countdown-birthday]");
const broadcastSummaryNode = document.querySelector("[data-countdown-broadcast]");
const liveStorageKey = "gyeonjahee-live-status";
const youtubeLatestNode = document.querySelector("[data-youtube-latest]");
const youtubeUpdatedNode = document.querySelector("[data-youtube-updated]");

const startOfDay = (date) =>
  new Date(date.getFullYear(), date.getMonth(), date.getDate());

const formatDate = (year, month, day) =>
  `${year}.${String(month).padStart(2, "0")}.${String(day).padStart(2, "0")}`;

const getUpcomingCountdown = (month, day, startYear) => {
  const today = startOfDay(new Date());
  let target = new Date(today.getFullYear(), month - 1, day);

  if (target < today) {
    target = new Date(today.getFullYear() + 1, month - 1, day);
  }

  const diffDays = Math.round((target - today) / 86400000);
  const cycle = target.getFullYear() - startYear;

  return {
    value: diffDays === 0 ? "D-Day" : `D-${diffDays}`,
    cycle,
    target,
  };
};

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

  ddayNode.textContent = diffDays === 0 ? "D-Day" : `D-${diffDays}`;
  cycleNode.textContent =
    anniversaryYear +
    "\uC8FC\uB144 \u00B7 " +
    formatDate(target.getFullYear(), month, day);
});

const birthdaySummary = getUpcomingCountdown(4, 28, 1993);
const broadcastSummary = getUpcomingCountdown(5, 11, 2012);

if (birthdaySummaryNode) {
  birthdaySummaryNode.textContent = birthdaySummary.value;
}

if (broadcastSummaryNode) {
  broadcastSummaryNode.textContent = broadcastSummary.value;
}

countdownChips.forEach((chip) => {
  const month = Number(chip.dataset.month);
  const day = Number(chip.dataset.day);
  const startYear = Number(chip.dataset.startYear);
  const mode = chip.dataset.mode;
  const valueNode = chip.querySelector(".countdown-value");
  const metaNode = chip.querySelector(".countdown-meta");

  if (!month || !day || !startYear || !mode || !valueNode || !metaNode) {
    return;
  }

  const today = startOfDay(new Date());
  const startDate = startOfDay(new Date(startYear, month - 1, day));

  if (mode === "elapsed") {
    const elapsedDays = Math.max(
      0,
      Math.round((today - startDate) / 86400000)
    );

    valueNode.textContent =
      elapsedDays.toLocaleString("ko-KR") + "\uC77C";
    metaNode.textContent =
      formatDate(startYear, month, day) +
      "\uBD80\uD130 \uC624\uB298\uAE4C\uC9C0";
    return;
  }

  let target = new Date(today.getFullYear(), month - 1, day);

  if (target < today) {
    target = new Date(today.getFullYear() + 1, month - 1, day);
  }

  const diffDays = Math.round((target - today) / 86400000);
  const cycle = target.getFullYear() - startYear;

  valueNode.textContent = diffDays === 0 ? "D-Day" : `D-${diffDays}`;
  metaNode.textContent =
    cycle + "\uC8FC\uB144 \u00B7 " + formatDate(target.getFullYear(), month, day);
});

const applyLiveStatus = (status) => {
  if (!liveStatusNode) {
    return;
  }

  const isOnAir = status === "onair";
  liveStatusNode.textContent = isOnAir
    ? "\uBC29\uC1A1 \uC911"
    : "\uC624\uD504\uB77C\uC778";
  liveStatusNode.classList.toggle("is-onair", isOnAir);
  liveStatusNode.classList.toggle("is-offline", !isOnAir);
};

const initialLiveStatus = localStorage.getItem(liveStorageKey) || "offline";
applyLiveStatus(initialLiveStatus);

if (liveToggleButton) {
  liveToggleButton.addEventListener("click", () => {
    const nextStatus =
      localStorage.getItem(liveStorageKey) === "onair" ? "offline" : "onair";

    localStorage.setItem(liveStorageKey, nextStatus);
    applyLiveStatus(nextStatus);
  });
}

const formatYouTubeDate = (value) => {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "날짜 정보 없음";
  }

  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
};

const trimText = (value, maxLength) => {
  if (!value) {
    return "";
  }

  return value.length > maxLength
    ? `${value.slice(0, maxLength).trim()}…`
    : value;
};

const renderYouTubeFallback = (message) => {
  if (!youtubeLatestNode) {
    return;
  }

  youtubeLatestNode.innerHTML = `
    <article class="youtube-latest-empty" data-reveal>
      <p>${message}</p>
    </article>
  `;

  const fallbackNode = youtubeLatestNode.querySelector("[data-reveal]");

  if (fallbackNode) {
    fallbackNode.classList.add("is-visible");
  }
};

const renderYouTubeLatest = (payload) => {
  if (!youtubeLatestNode) {
    return;
  }

  const items = Array.isArray(payload?.items) ? payload.items.slice(0, 3) : [];

  if (!items.length) {
    renderYouTubeFallback("최신 영상 데이터가 아직 준비되지 않았습니다.");
    return;
  }

  youtubeLatestNode.innerHTML = items
    .map(
      (item) => `
        <a class="youtube-latest-card" href="${item.url}" target="_blank" rel="noreferrer" data-reveal>
          <div class="youtube-latest-thumb">
            <img src="${item.thumbnail}" alt="${item.title} 썸네일">
            <span class="youtube-latest-badge">최신 업로드</span>
          </div>
          <div class="youtube-latest-body">
            <h4>${item.title}</h4>
            <p>${trimText(item.description, 88) || "영상 설명은 유튜브에서 바로 확인할 수 있습니다."}</p>
          </div>
          <div class="youtube-latest-foot">
            <span>${formatYouTubeDate(item.published_at)}</span>
            <span>유튜브에서 보기</span>
          </div>
        </a>
      `
    )
    .join("");

  youtubeLatestNode.querySelectorAll("[data-reveal]").forEach((card) => {
    card.classList.add("is-visible");
  });

  if (youtubeUpdatedNode) {
    youtubeUpdatedNode.textContent = payload?.updated_at
      ? `마지막 갱신 ${formatYouTubeDate(payload.updated_at)}`
      : "최신 업로드 기준으로 정리했습니다.";
  }
};

const loadYouTubeLatest = async () => {
  if (!youtubeLatestNode) {
    return;
  }

  try {
    const response = await fetch(`data/youtube-latest.json?ts=${Date.now()}`, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    renderYouTubeLatest(payload);
  } catch (error) {
    renderYouTubeFallback("최신 영상을 불러오지 못했습니다. 잠시 뒤 다시 확인해 주세요.");

    if (youtubeUpdatedNode) {
      youtubeUpdatedNode.textContent = "자동 갱신 연결을 확인하는 중입니다.";
    }
  }
};

loadYouTubeLatest();

document.body.classList.remove("light-theme");
localStorage.removeItem("gyeonjahee-theme");
