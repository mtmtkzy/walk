/* ============================================================
   Phase1: 「動くと冒険が進む」を確認するための最小プロトタイプ
   構成:
     1. 定数・DOM要素の取得
     2. 状態を保持する変数
     3. MediaPipe Pose の初期化とカメラ起動
     4. 検出結果を受け取ったときの処理(骨格描画・判定呼び出し)
     5. 足踏み(歩行)判定ロジック
     5b. 左右移動(3レーン)判定ロジック
     5c. 障害物(岩)の生成・回避/衝突判定ロジック
     6. ゲーム画面(背景スクロール・キャラクター・距離・障害物)の描画ループ
     7. 効果音ユーティリティ
     8. 起動処理
   ============================================================ */


/* ------------------------------------------------------------
   1. 定数・DOM要素の取得
   ------------------------------------------------------------ */

// MediaPipe Poseのランドマーク番号(必要な部位のみ抜粋)
// 参考: https://google.github.io/mediapipe/solutions/pose.html
const LANDMARK = {
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
};

// 足踏み判定の閾値(ヒステリシス方式でチラつきを防ぐ)
// legLift = 股関節のY座標 - 膝のY座標
// 値が大きい(上がる)ほど「膝が上がっている」状態を意味する
const LIFT_ON_THRESHOLD = -0.12;   // これを超えたら「脚が上がった」と判定
const LIFT_OFF_THRESHOLD = -0.18;  // これを下回ったら「脚が下りた」と判定(1歩完了)

// ランドマークの信頼度がこれ未満の場合は判定に使わない(誤検出防止)
const VISIBILITY_THRESHOLD = 0.5;

// 1歩ごとに加算するスクロール速度(px/秒)
const SPEED_BOOST = 260;
// スクロール速度の減衰率(何もしないと徐々に止まる)
const SPEED_DECAY_PER_SEC = 0.85;
// 距離表示のスケール(スクロールpx数を「見かけ上の距離(m)」に変換する係数)
const DISTANCE_SCALE = 0.05;

// --- 左右移動(3レーン)関連の定数 ---

// レーンの番号(左 / 中央 / 右)
const LANE = { LEFT: -1, CENTER: 0, RIGHT: 1 };

// 腰のX座標(0〜1に正規化)の生値をなめらかにするための指数移動平均係数
// 値が小さいほど、より滑らかになる(=反応はやや遅くなる)
const HIP_X_SMOOTHING = 0.25;

// レーン切り替えの境界(画面を横に3分割する位置)
const LANE_BOUNDARY_LEFT = 1 / 3;
const LANE_BOUNDARY_RIGHT = 2 / 3;

// 境界付近でのチラつき防止用マージン(ヒステリシス)
// 中央→端 へは境界の外側まで、端→中央 へは境界の内側まで動かないと切り替わらない
const LANE_HYSTERESIS = 0.06;

// キャラクターがレーンを切り替えるときのイージング速度(大きいほど速く追従)
const LANE_EASE_RATE = 8;

// 画面上でキャラクターが左右に動く最大幅(px)。道の手前側の幅を基準に決める
const LANE_OFFSET_PX = 210;

// レーン番号 → 道幅に対する横方向の比率(区切り線の位置(±1/6)と整合させ、
// 各レーンの中心が来るように -1/3, 0, +1/3 としている)
const LANE_FRACTION = {
  [LANE.LEFT]: -1 / 3,
  [LANE.CENTER]: 0,
  [LANE.RIGHT]: 1 / 3,
};

// --- 障害物(岩)関連の定数 ---

// 岩が地平線付近に出現してから、プレイヤーの位置まで到達するのにかかる「世界距離」
// (道の継ぎ目と同じ考え方。値を大きくするほど、出現から到達までの時間的猶予が増える)
const OBSTACLE_SPAWN_DISTANCE = 820;

// 次の岩が出現するまでの間隔(世界距離、ランダムに変化させて単調にならないようにする)
const OBSTACLE_SPAWN_INTERVAL_MIN = 300;
const OBSTACLE_SPAWN_INTERVAL_MAX = 460;

// この深さ(t)まで岩が到達したら、回避できたか/ぶつかったかを判定する
// (プレイヤーの表示位置とほぼ同じ深さになったタイミング)
const OBSTACLE_RESOLVE_T = 0.94;

// 岩を通り過ぎた後、画面から消すまでの深さ(1を超えても少し描画し続け、自然に消える)
const OBSTACLE_REMOVE_T = 1.15;

// ぶつかったときの減速率(完全停止はさせず、少し勢いが落ちる程度にとどめる)
const OBSTACLE_HIT_SLOWDOWN = 0.4;

// 衝突/回避の演出(色の変化など)を表示する時間(秒)
const FEEDBACK_FLASH_DURATION = 0.45;

// DOM要素の取得
const videoElement = document.getElementById("inputVideo");
const debugCanvas = document.getElementById("debugCanvas");
const debugCtx = debugCanvas.getContext("2d");
const gameCanvas = document.getElementById("gameCanvas");
const gameCtx = gameCanvas.getContext("2d");
const statusText = document.getElementById("statusText");


/* ------------------------------------------------------------
   2. 状態を保持する変数
   ------------------------------------------------------------ */

// 現在、脚が「上がっている」か「下りている」か(ヒステリシス判定用)
let legState = "down";

// 検出できた歩数(足踏み回数)
let stepCount = 0;

// 現在のスクロール速度(px/秒)。歩くたびに増え、時間とともに減衰する
let scrollSpeed = 0;

// 背景のスクロール量(累積)
let scrollOffset = 0;

// 見かけ上の移動距離(表示用、単位はm)
let distance = 0;

// 直近で人体が検出できているかどうか
let isPersonDetected = false;

// アニメーション用の経過時間(キャラクターのバウンド演出に使用)
let elapsedTime = 0;

// 腰のX座標(鏡映し後、0〜1)をなめらかにした値。まだ計算していない場合はnull
let smoothedHipX = null;

// 現在判定されているレーン(LANE.LEFT / CENTER / RIGHT)
let currentLane = LANE.CENTER;

// キャラクターの見た目上の左右位置(-1〜1の連続値)。
// currentLaneの値へ毎フレーム少しずつ近づけることで、瞬間移動ではなく
// 「すーっと横に移動する」なめらかな動きになる
let characterLaneT = 0;

// --- 障害物(岩)関連の状態 ---

// 現在画面上に存在する岩のリスト
// 各要素: { resolveAtScroll: 到達判定を行うscrollOffsetの値, lane: 出現レーン, resolved: 判定済みか, t: 現在の深さ(描画用) }
let obstacles = [];

// 次の岩を出現させるscrollOffsetのしきい値(最初の岩は少し助走をつけてから出す)
let nextObstacleSpawnAt = 220;

// 回避に成功した回数(ポジティブな指標として表示する)
let avoidedCount = 0;

// 衝突演出・回避演出の残り表示時間(秒)。0より大きい間だけ演出色を出す
let hitFlashTimer = 0;
let avoidedFlashTimer = 0;


/* ------------------------------------------------------------
   3. MediaPipe Pose の初期化とカメラ起動
   ------------------------------------------------------------ */

// Poseインスタンスを作成。CDNから読み込んだモデルファイルの場所を指定する
const pose = new Pose({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
});

// Poseの検出設定
pose.setOptions({
  modelComplexity: 1,        // 精度と速度のバランス(0〜2、1が標準)
  smoothLandmarks: true,      // フレーム間の座標をなめらかにする(MediaPipe内蔵の平滑化)
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

// 毎フレーム、検出結果が返ってくるとこの関数が呼ばれる
pose.onResults(onPoseResults);

// カメラ映像を取得し、1フレームごとにPoseへ渡すためのCameraユーティリティ
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await pose.send({ image: videoElement });
  },
  width: 640,
  height: 480,
});


/* ------------------------------------------------------------
   4. 検出結果を受け取ったときの処理
   ------------------------------------------------------------ */

function onPoseResults(results) {
  // --- デバッグ用キャンバスにカメラ映像+骨格を描画 ---
  debugCtx.save();
  debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);

  // 映像を左右反転して描画(鏡のように、自分の動きと同じ向きに見せるため)
  debugCtx.translate(debugCanvas.width, 0);
  debugCtx.scale(-1, 1);
  debugCtx.drawImage(results.image, 0, 0, debugCanvas.width, debugCanvas.height);

  if (results.poseLandmarks) {
    isPersonDetected = true;

    // 骨格の線と点を描画(drawing_utils.jsが提供するグローバル関数)
    drawConnectors(debugCtx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: "#2f6f6f",
      lineWidth: 3,
    });
    drawLandmarks(debugCtx, results.poseLandmarks, {
      color: "#ff8c42",
      lineWidth: 1,
      radius: 3,
    });

    // 足踏み判定ロジックへ渡す
    detectStepping(results.poseLandmarks);

    // 左右移動(レーン)判定ロジックへ渡す
    updateLanePosition(results.poseLandmarks);
  } else {
    isPersonDetected = false;
  }

  debugCtx.restore();

  // --- 状態表示テキストの更新 ---
  updateStatusText();
}

function updateStatusText() {
  if (!isPersonDetected) {
    statusText.textContent = "からだが みえません(カメラの前に立ってね)";
    return;
  }

  // レーン名(体を動かさなくても、今どのレーンと判定されているか確認できるようにする)
  const laneLabel =
    currentLane === LANE.LEFT ? "ひだり" : currentLane === LANE.RIGHT ? "みぎ" : "まんなか";

  if (scrollSpeed > 5) {
    statusText.textContent = `すすんでいます! いま: ${laneLabel}`;
  } else {
    statusText.textContent = `その ばで あしぶみ してみよう(いま: ${laneLabel})`;
  }
}


/* ------------------------------------------------------------
   5. 足踏み(歩行)判定ロジック
   ------------------------------------------------------------
   考え方:
     ・股関節(hip)と膝(knee)のY座標の差(legLift)を左右それぞれ計算
     ・値が大きいほど「膝が持ち上がっている」ことを意味する
     ・左右のうち、より高く上がっている方の値を採用する
       (片足ずつ交互に上がる、足踏み・歩行の動きを捉えるため)
     ・「上がった→下りた」の1サイクルを検出したら1歩とカウントする
     ・ON/OFFで別々の閾値を使う(ヒステリシス)ことで、
       境界付近での小さなブレによる誤検出を防いでいる
   ------------------------------------------------------------ */

function detectStepping(landmarks) {
  const leftHip = landmarks[LANDMARK.LEFT_HIP];
  const rightHip = landmarks[LANDMARK.RIGHT_HIP];
  const leftKnee = landmarks[LANDMARK.LEFT_KNEE];
  const rightKnee = landmarks[LANDMARK.RIGHT_KNEE];

  // 必要な部位の信頼度が低い場合は判定をスキップ(下半身が映っていない等)
  const allVisible =
    leftHip.visibility > VISIBILITY_THRESHOLD &&
    rightHip.visibility > VISIBILITY_THRESHOLD &&
    leftKnee.visibility > VISIBILITY_THRESHOLD &&
    rightKnee.visibility > VISIBILITY_THRESHOLD;

  if (!allVisible) {
    return; // 下半身が見えていないので判定しない(誤検出防止)
  }

  // legLift: 股関節Y座標 - 膝Y座標(値が大きいほど脚が高く上がっている)
  const leftLegLift = leftHip.y - leftKnee.y;
  const rightLegLift = rightHip.y - rightKnee.y;

  // 左右どちらか、より高く上がっている方を採用
  const maxLegLift = Math.max(leftLegLift, rightLegLift);

  if (legState === "down" && maxLegLift > LIFT_ON_THRESHOLD) {
    // 脚が持ち上がった(まだカウントはしない。下りたときに1歩とする)
    legState = "up";
  } else if (legState === "up" && maxLegLift < LIFT_OFF_THRESHOLD) {
    // 脚が下りた → 1歩分の動きが完了したとみなす
    legState = "down";
    onStepDetected();
  }
}

function onStepDetected() {
  stepCount += 1;

  // 1歩ごとにスクロール速度を加算(すでに動いている場合は上乗せされる)
  scrollSpeed += SPEED_BOOST;
}


/* ------------------------------------------------------------
   5b. 左右移動(3レーン)判定ロジック
   ------------------------------------------------------------
   考え方:
     ・左右の股関節のX座標を平均し、「腰の中心位置」を求める
     ・カメラ画像は反転していない(鏡ではない)ため、デバッグ表示と
       向きを合わせるために 1 - x で鏡映しに変換する
       (これにより「自分が右に動けば画面でも右に動く」という
        直感的な対応になる)
     ・値をそのまま使うと細かいブレで判定が揺れるため、
       指数移動平均(EMA)で毎フレーム少しずつなめらかにする
     ・画面を横に3分割し、なめらかにした値がどの範囲にあるかでレーンを決める
     ・レーン切り替えには足踏み判定と同様にヒステリシスを用い、
       境界付近に立ち続けたときのチラつきを防ぐ
   ------------------------------------------------------------ */

function updateLanePosition(landmarks) {
  const leftHip = landmarks[LANDMARK.LEFT_HIP];
  const rightHip = landmarks[LANDMARK.RIGHT_HIP];

  // 股関節の信頼度が低い場合は判定しない(誤検出防止)
  if (leftHip.visibility < VISIBILITY_THRESHOLD || rightHip.visibility < VISIBILITY_THRESHOLD) {
    return;
  }

  // 腰の中心のX座標(カメラ画像そのままの座標系。0=画像の左端、1=画像の右端)
  const hipCenterXRaw = (leftHip.x + rightHip.x) / 2;

  // 鏡映しに変換(デバッグ映像や実際の見え方と直感を合わせるため)
  const mirroredX = 1 - hipCenterXRaw;

  // 指数移動平均でなめらかにする(初回はそのまま採用)
  if (smoothedHipX === null) {
    smoothedHipX = mirroredX;
  } else {
    smoothedHipX += (mirroredX - smoothedHipX) * HIP_X_SMOOTHING;
  }

  // なめらかにした値から、ヒステリシス付きでレーンを決定する
  currentLane = decideLane(smoothedHipX, currentLane);
}

function decideLane(x, lane) {
  if (lane === LANE.CENTER) {
    if (x < LANE_BOUNDARY_LEFT - LANE_HYSTERESIS) return LANE.LEFT;
    if (x > LANE_BOUNDARY_RIGHT + LANE_HYSTERESIS) return LANE.RIGHT;
    return LANE.CENTER;
  }
  if (lane === LANE.LEFT) {
    // 中央側に十分戻ってきたら中央レーンへ
    return x > LANE_BOUNDARY_LEFT + LANE_HYSTERESIS ? LANE.CENTER : LANE.LEFT;
  }
  if (lane === LANE.RIGHT) {
    return x < LANE_BOUNDARY_RIGHT - LANE_HYSTERESIS ? LANE.CENTER : LANE.RIGHT;
  }
  return LANE.CENTER;
}


/* ------------------------------------------------------------
   5c. 障害物(岩)の生成・回避/衝突判定ロジック
   ------------------------------------------------------------
   考え方:
     ・岩は仕様どおり必ず「中央レーン」に出現する
     ・道の継ぎ目(枕木)と同じ仕組みで、scrollOffsetを基準に
       「奥(地平線)→手前(プレイヤー)」へ近づいてくるように動かす
     ・岩がプレイヤーとほぼ同じ深さ(OBSTACLE_RESOLVE_T)まで来た瞬間に、
       「今どのレーンにいるか(currentLane)」と岩のレーンを比較して
       避けられたかどうかを1回だけ判定する
     ・ぶつかった場合も、完全停止や強い否定表現は使わず、
       少し減速する程度の穏やかなフィードバックにとどめる
       (身体を動かすこと自体をネガティブに感じさせないため)
   ------------------------------------------------------------ */

// scrollOffsetの進み具合を見て、新しい岩を出現させるべきタイミングかどうかを判定する
function spawnObstacleIfNeeded() {
  if (scrollOffset < nextObstacleSpawnAt) {
    return;
  }

  obstacles.push({
    // この値にscrollOffsetが追いついたとき、岩がプレイヤーの位置に到達したとみなす
    resolveAtScroll: scrollOffset + OBSTACLE_SPAWN_DISTANCE,
    lane: LANE.CENTER, // 仕様により、岩は必ず中央レーンに出現する
    resolved: false,
    t: 0,
  });

  // 次の岩の出現タイミングをランダムに決めておく(単調な繰り返しを避ける)
  const interval =
    OBSTACLE_SPAWN_INTERVAL_MIN +
    Math.random() * (OBSTACLE_SPAWN_INTERVAL_MAX - OBSTACLE_SPAWN_INTERVAL_MIN);
  nextObstacleSpawnAt = scrollOffset + interval;
}

// 全ての岩について、現在の深さ(t)を更新し、判定タイミングに達したものを判定する
function updateObstacles() {
  for (const obs of obstacles) {
    const remaining = obs.resolveAtScroll - scrollOffset; // プレイヤーまでの残り距離
    obs.t = 1 - remaining / OBSTACLE_SPAWN_DISTANCE; // 0=地平線, 1=プレイヤーの位置

    if (!obs.resolved && obs.t >= OBSTACLE_RESOLVE_T) {
      obs.resolved = true;

      if (currentLane === obs.lane) {
        onObstacleHit();
      } else {
        onObstacleAvoided();
      }
    }
  }

  // 画面を通り過ぎて見えなくなった岩は配列から取り除く(メモリ・描画のムダを防ぐ)
  obstacles = obstacles.filter((obs) => obs.t < OBSTACLE_REMOVE_T);
}

function onObstacleHit() {
  // 完全に止めるのではなく、少し勢いが落ちる程度の穏やかな反応にする
  scrollSpeed *= OBSTACLE_HIT_SLOWDOWN;
  hitFlashTimer = FEEDBACK_FLASH_DURATION;
  playTone(220, 0.18, "sine"); // 低めの、驚かせすぎない音
}

function onObstacleAvoided() {
  avoidedCount += 1;
  avoidedFlashTimer = FEEDBACK_FLASH_DURATION;
  playTone(880, 0.18, "sine"); // 明るく高めの、成功を感じさせる音
}


/* ------------------------------------------------------------
   6. ゲーム画面の描画ループ
   ------------------------------------------------------------
   ・requestAnimationFrameで毎フレーム呼ばれる
   ・スクロール速度は時間とともに減衰させ、
     「歩き続けないと進まない」という体験にする
   ------------------------------------------------------------ */

let lastTimestamp = null;

function gameLoop(timestamp) {
  if (lastTimestamp === null) {
    lastTimestamp = timestamp;
  }
  const dt = (timestamp - lastTimestamp) / 1000; // 秒単位の経過時間
  lastTimestamp = timestamp;
  elapsedTime += dt;

  // --- スクロール速度の減衰(指数減衰) ---
  scrollSpeed *= Math.pow(SPEED_DECAY_PER_SEC, dt * 60);
  if (scrollSpeed < 1) {
    scrollSpeed = 0;
  }

  // --- スクロール量・距離の更新 ---
  scrollOffset += scrollSpeed * dt;
  distance += scrollSpeed * dt * DISTANCE_SCALE;

  // --- 障害物(岩)の出現・接近・判定 ---
  spawnObstacleIfNeeded();
  updateObstacles();

  // --- 衝突/回避の演出タイマーを減衰させる ---
  hitFlashTimer = Math.max(0, hitFlashTimer - dt);
  avoidedFlashTimer = Math.max(0, avoidedFlashTimer - dt);

  // --- キャラクターの見た目上の左右位置を、判定されたレーンへ少しずつ近づける ---
  // (瞬間移動ではなく、なめらかにスライドさせることで動きが分かりやすくなる)
  characterLaneT += (currentLane - characterLaneT) * Math.min(1, LANE_EASE_RATE * dt);

  drawGame();

  requestAnimationFrame(gameLoop);
}

/* ------------------------------------------------------------
   奥行き表現の設定
   ------------------------------------------------------------
   ・キャラクターは画面下中央に固定し、後ろ姿で「画面の奥」に向かって走る
   ・実際に動くのはキャラクターではなく、道・目印・景色の方
   ・手前ほど大きく速く、奥(地平線)ほど小さくゆっくり見えるようにして
     遠近感(パース)を表現する
   ------------------------------------------------------------ */

// 地平線(道が収束して見える高さの割合)と、画面下端(道が最も広がる位置)
const HORIZON_RATIO = 0.38;
const GROUND_BOTTOM_RATIO = 1.0;

// 道の幅(地平線での幅 / 手前での幅)
const ROAD_WIDTH_TOP = 40;
const ROAD_WIDTH_BOTTOM = 720;

// 道の継ぎ目(枕木のような横線)を奥から手前へ流すための設定
const TIE_SPACING = 90;      // 継ぎ目同士の世界上の間隔
const TIE_TOTAL_DEPTH = 900; // 継ぎ目が奥から手前まで移動する総距離(この値で1周期)
const TIE_COUNT = Math.ceil(TIE_TOTAL_DEPTH / TIE_SPACING) + 1;

// 道の脇に置く目印(木)の設定。左右交互に配置し、道の奥行きと連動させる
const DECOR_SPACING = 260;
const DECOR_TOTAL_DEPTH = TIE_TOTAL_DEPTH;
const DECOR_COUNT = Math.ceil(DECOR_TOTAL_DEPTH / DECOR_SPACING) + 1;

// 0〜1の深さ割合(t: 0=地平線の奥, 1=手前)から、実際のY座標を計算する
// 単純な線形補間ではなく、tを2乗することで
// 「近いものほど急に大きく・速く見える」遠近感の効果を出している
function depthToScreenY(t, h) {
  const horizonY = h * HORIZON_RATIO;
  const bottomY = h * GROUND_BOTTOM_RATIO;
  return horizonY + (bottomY - horizonY) * (t * t);
}

function depthToRoadWidth(t) {
  return ROAD_WIDTH_TOP + (ROAD_WIDTH_BOTTOM - ROAD_WIDTH_TOP) * (t * t);
}

function drawGame() {
  const w = gameCanvas.width;
  const h = gameCanvas.height;

  // --- 空(背景) ---
  gameCtx.fillStyle = "#cdeaf0";
  gameCtx.fillRect(0, 0, w, h);

  // --- 遠くの丘(地平線の少し上に固定。奥行きの目安として使う) ---
  drawHills(w, h);

  // --- 地面(草原) ---
  const horizonY = h * HORIZON_RATIO;
  gameCtx.fillStyle = "#bfe3a0";
  gameCtx.fillRect(0, horizonY, w, h - horizonY);

  // --- 道(奥から手前に向かって広がる台形) + 継ぎ目の流れ ---
  drawRoad(w, h);

  // --- 道の脇の目印(木)。奥から手前へ流れることで奥行き移動を強調する ---
  drawRoadsideMarkers(w, h);

  // --- 障害物(岩)。奥から近づいてきて、プレイヤーの手前まで来ると回避/衝突が判定される ---
  drawObstacles(w, h);

  // --- キャラクター(後ろ姿・画面下中央に固定。歩いているときだけ手足を動かす) ---
  drawCharacterFromBehind(w, h);

  // --- 距離表示(ゲーム画面左上に大きく表示) ---
  drawDistanceHUD();
}

function drawHills(w, h) {
  const horizonY = h * HORIZON_RATIO;
  gameCtx.fillStyle = "#a9dede";
  gameCtx.beginPath();
  gameCtx.ellipse(w * 0.25, horizonY, 220, 50, 0, 0, Math.PI * 2);
  gameCtx.fill();
  gameCtx.beginPath();
  gameCtx.ellipse(w * 0.75, horizonY, 260, 60, 0, 0, Math.PI * 2);
  gameCtx.fill();
}

function drawRoad(w, h) {
  const horizonY = h * HORIZON_RATIO;
  const bottomY = h * GROUND_BOTTOM_RATIO;
  const centerX = w / 2;

  // 道本体(奥は細く、手前は広い台形)
  gameCtx.fillStyle = "#e4d9b8";
  gameCtx.beginPath();
  gameCtx.moveTo(centerX - ROAD_WIDTH_TOP / 2, horizonY);
  gameCtx.lineTo(centerX + ROAD_WIDTH_TOP / 2, horizonY);
  gameCtx.lineTo(centerX + ROAD_WIDTH_BOTTOM / 2, bottomY);
  gameCtx.lineTo(centerX - ROAD_WIDTH_BOTTOM / 2, bottomY);
  gameCtx.closePath();
  gameCtx.fill();

  // --- レーンの区切り線(左/中央/右の3分割を目で見て分かるようにする) ---
  // 道の左右の輪郭線と同じく直線(台形)で描く。道幅を3等分する位置に2本引く
  gameCtx.strokeStyle = "#d8c99a";
  gameCtx.lineWidth = 2;
  gameCtx.setLineDash([10, 10]);
  for (const frac of [-1 / 6, 1 / 6]) {
    gameCtx.beginPath();
    gameCtx.moveTo(centerX + frac * ROAD_WIDTH_TOP, horizonY);
    gameCtx.lineTo(centerX + frac * ROAD_WIDTH_BOTTOM, bottomY);
    gameCtx.stroke();
  }
  gameCtx.setLineDash([]);

  // 道の継ぎ目(枕木状の線)を、奥から手前へ流れるように描画
  // scrollOffsetに応じて各継ぎ目の「奥からの距離」を計算し、
  // 手前に来るほど(tが大きいほど)太く・間隔広めに見せることで奥行き移動を表現する
  gameCtx.strokeStyle = "#cbbd8f";
  gameCtx.lineCap = "round";

  for (let i = 0; i < TIE_COUNT; i++) {
    // このタイル(継ぎ目)の「奥からの距離」を、スクロール量を使って計算
    // %演算とTIE_TOTAL_DEPTHでループさせることで、無限に奥から流れてくるように見せる
    const raw = (i * TIE_SPACING - scrollOffset) % TIE_TOTAL_DEPTH;
    const distFromViewer = ((raw % TIE_TOTAL_DEPTH) + TIE_TOTAL_DEPTH) % TIE_TOTAL_DEPTH;
    const t = 1 - distFromViewer / TIE_TOTAL_DEPTH; // t=1: 目の前、t=0: 地平線

    if (t <= 0) continue;

    const y = depthToScreenY(t, h);
    const roadWidth = depthToRoadWidth(t);
    gameCtx.lineWidth = 2 + t * 8;
    gameCtx.beginPath();
    gameCtx.moveTo(centerX - roadWidth / 2, y);
    gameCtx.lineTo(centerX + roadWidth / 2, y);
    gameCtx.stroke();
  }
}

function drawRoadsideMarkers(w, h) {
  const centerX = w / 2;

  for (let i = 0; i < DECOR_COUNT; i++) {
    const raw = (i * DECOR_SPACING - scrollOffset) % DECOR_TOTAL_DEPTH;
    const distFromViewer = ((raw % DECOR_TOTAL_DEPTH) + DECOR_TOTAL_DEPTH) % DECOR_TOTAL_DEPTH;
    const t = 1 - distFromViewer / DECOR_TOTAL_DEPTH;

    if (t <= 0.02) continue;

    const y = depthToScreenY(t, h);
    const roadWidth = depthToRoadWidth(t);
    const side = i % 2 === 0 ? -1 : 1; // 左右交互に配置
    const x = centerX + side * (roadWidth / 2 + 30 + t * 40);
    const size = 14 + t * 46; // 手前ほど大きく見せる

    gameCtx.font = `${size}px sans-serif`;
    gameCtx.textAlign = "center";
    gameCtx.textBaseline = "bottom";
    gameCtx.fillText("🌳", x, y);
  }
}

function drawObstacles(w, h) {
  const centerX = w / 2;

  for (const obs of obstacles) {
    const t = Math.max(0, Math.min(1.1, obs.t));
    if (t <= 0.02) continue;

    const y = depthToScreenY(t, h);
    const roadWidth = depthToRoadWidth(t);
    // 岩のレーン(必ず中央)に応じて、道幅に対する横位置を決める
    const x = centerX + LANE_FRACTION[obs.lane] * roadWidth;
    const size = 20 + t * 70; // 手前に来るほど大きく見せ、接近を分かりやすくする

    gameCtx.font = `${size}px sans-serif`;
    gameCtx.textAlign = "center";
    gameCtx.textBaseline = "bottom";
    gameCtx.fillText("🪨", x, y);
  }
}

function drawCharacterFromBehind(w, h) {
  const groundBottomY = h * GROUND_BOTTOM_RATIO;
  const centerX = w / 2;
  const baseY = groundBottomY - 30; // 画面下からの固定位置(奥へは動かない)

  // 動いているときだけ手足を交互に振るアニメーション(足踏み・走行の表現)
  const isMoving = scrollSpeed > 5;
  const swing = isMoving ? Math.sin(elapsedTime * 12) : 0; // -1〜1で振れる
  const bounceHeight = isMoving ? Math.abs(Math.sin(elapsedTime * 12)) * 8 : 0;

  const hipY = baseY - bounceHeight;
  // characterLaneT(-1〜1のなめらかな値)を、実際の画面上の左右オフセットに変換
  const laneOffsetX = characterLaneT * LANE_OFFSET_PX;
  const headRadius = 20;
  const bodyHeight = 46;
  const bodyWidth = 30;

  gameCtx.save();
  gameCtx.translate(centerX + laneOffsetX, hipY);

  // --- 脚(後ろ姿なので左右対称に交互に振れる) ---
  gameCtx.strokeStyle = "#4a5759";
  gameCtx.lineWidth = 10;
  gameCtx.lineCap = "round";

  gameCtx.beginPath();
  gameCtx.moveTo(-10, 0);
  gameCtx.lineTo(-10 + swing * 14, 42);
  gameCtx.stroke();

  gameCtx.beginPath();
  gameCtx.moveTo(10, 0);
  gameCtx.lineTo(10 - swing * 14, 42);
  gameCtx.stroke();

  // --- 胴体(後ろ姿なので、ゼッケンのような目印を背中につけて向きを分かりやすくする) ---
  // 通常は落ち着いた色。ぶつかった直後は少し警告色に、避けられた直後はお祝いの金色に変える
  gameCtx.fillStyle =
    hitFlashTimer > 0 ? "#e0765a" : avoidedFlashTimer > 0 ? "#ffcf5c" : "#4c8577";
  gameCtx.beginPath();
  gameCtx.roundRect(-bodyWidth / 2, -bodyHeight, bodyWidth, bodyHeight, 12);
  gameCtx.fill();

  gameCtx.fillStyle = "#ffffff";
  gameCtx.font = "bold 16px sans-serif";
  gameCtx.textAlign = "center";
  gameCtx.textBaseline = "middle";
  gameCtx.fillText("↑", 0, -bodyHeight / 2);

  // --- 頭(髪の毛の位置などは付けず、シンプルな丸のみ。後ろ姿として認識しやすくする) ---
  gameCtx.fillStyle = "#f2c9a0";
  gameCtx.beginPath();
  gameCtx.arc(0, -bodyHeight - headRadius + 6, headRadius, 0, Math.PI * 2);
  gameCtx.fill();

  // --- 回避に成功した直後だけ、キャラクターの周りにきらめきを表示する ---
  if (avoidedFlashTimer > 0) {
    drawSparkles(avoidedFlashTimer / FEEDBACK_FLASH_DURATION, bodyHeight, headRadius);
  }

  gameCtx.restore();
}

// 回避成功時のきらめき演出。progressは1(発生直後)→0(消える直前)へ変化する
function drawSparkles(progress, bodyHeight, headRadius) {
  const sparkleCount = 6;
  const radius = 46;
  gameCtx.save();
  gameCtx.globalAlpha = progress;
  gameCtx.fillStyle = "#fff3b0";
  for (let i = 0; i < sparkleCount; i++) {
    const angle = (i / sparkleCount) * Math.PI * 2 + progress * 2; // 少し回転させて動きを出す
    const sx = Math.cos(angle) * radius;
    const sy = -bodyHeight - headRadius + Math.sin(angle) * radius * 0.6;
    gameCtx.beginPath();
    gameCtx.arc(sx, sy, 4, 0, Math.PI * 2);
    gameCtx.fill();
  }
  gameCtx.restore();
}

function drawDistanceHUD() {
  gameCtx.font = "bold 28px sans-serif";
  gameCtx.textAlign = "left";
  gameCtx.textBaseline = "top";
  gameCtx.fillStyle = "#2f6f6f";
  gameCtx.fillText(`きょり: ${distance.toFixed(1)} m`, 20, 20);

  // 回避数は失敗を数えない、ポジティブな指標としてだけ表示する
  gameCtx.font = "bold 22px sans-serif";
  gameCtx.fillText(`よけた いわ: ${avoidedCount}`, 20, 56);
}


/* ------------------------------------------------------------
   7. 効果音ユーティリティ
   ------------------------------------------------------------
   ・外部の音声ファイルを使わず、Web Audio APIで短いトーン(ビープ音)を
     その場で生成して鳴らす(仕組みがシンプルで、音量・音程の調整もしやすい)
   ・音量は控えめにし、驚かせるような大きい音・急な音は避ける
   ・ブラウザによっては「ユーザー操作前は音声再生がブロックされる」仕様があるため、
     最初の再生タイミングでAudioContextを作成・再開する形にしている
   ------------------------------------------------------------ */

let audioCtx = null;

function playTone(frequency, duration, waveType = "sine") {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }

    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.type = waveType;
    oscillator.frequency.value = frequency;

    // 音量は控えめに設定し、終わり際に自然に減衰させる(プツッと切れる違和感を防ぐ)
    gainNode.gain.value = 0.15;
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + duration);
  } catch (error) {
    // 音声再生に失敗しても、ゲーム自体は継続できるようにする(致命的エラーにしない)
    console.warn("効果音の再生に失敗しました:", error);
  }
}


/* ------------------------------------------------------------
   8. 起動処理
   ------------------------------------------------------------ */

async function start() {
  statusText.textContent = "カメラを起動しています…";
  try {
    await camera.start();
    statusText.textContent = "からだが みえません(カメラの前に立ってね)";
    requestAnimationFrame(gameLoop);
  } catch (error) {
    // カメラ権限が拒否された場合などのエラーハンドリング
    console.error("カメラの起動に失敗しました:", error);
    statusText.textContent = "カメラを起動できませんでした。権限設定を確認してください。";
  }
}

start();
