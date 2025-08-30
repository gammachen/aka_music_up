<!-- _coverpage.md封面设置 -->


<p align="center">
    <img src="logo.png" width="150"/>
</p>

<h1 align="center">熵策咨询</h1>

> 您的数字化转型伙伴

<style>
.feature-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 20px;
    margin-top: 40px;
}
.feature-item {
    width: 200px;
    text-align: center;
    border: 1px solid #eee;
    padding: 15px;
    border-radius: 8px;
    transition: all 0.3s;
}
.feature-item:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transform: translateY(-5px);
}
.feature-item img {
    width: 100%;
    height: 120px;
    object-fit: cover;
    border-radius: 4px;
}
.feature-item p {
    margin-top: 10px;
    font-weight: bold;
    color: #333;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.tab-switcher {
  display: flex;
  justify-content: center;
  gap: 0;
  margin: 40px 0 0 0;
}
.tab-btn {
  flex: 1;
  max-width: 340px;
  font-size: 1.4em;
  font-weight: bold;
  padding: 28px 0;
  border: none;
  outline: none;
  cursor: pointer;
  border-radius: 16px 16px 0 0;
  background: #1a3a5d;
  color: #fff;
  transition: background 0.3s, color 0.3s;
  position: relative;
  z-index: 2;
}
.tab-btn.active, .tab-btn:hover {
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  color: #003a5d;
}
.tab-content-area {
  background: #12263a;
  border-radius: 0 0 24px 24px;
  margin: 0 auto 32px auto;
  max-width: 900px;
  box-shadow: 0 2px 16px rgba(0,255,231,0.06);
  padding: 36px 32px 32px 32px;
  min-height: 260px;
  color: #fff;
  display: flex;
  flex-wrap: wrap;
  gap: 32px;
  align-items: flex-start;
}
.tab-panel { display: none; width: 100%; }
.tab-panel.active { display: flex; }
.tab-panel .tab-info {
  flex: 1;
  min-width: 260px;
}
.tab-panel .tab-info h3 {
  font-size: 1.3em;
  font-weight: bold;
  margin: 0 0 10px 0;
  display: flex;
  align-items: center;
}
.tab-panel .tab-info .icon {
  width: 38px;
  height: 38px;
  margin-right: 12px;
  vertical-align: middle;
}
.tab-panel .tab-info p {
  color: #b6eaff;
  font-size: 1em;
  margin-bottom: 18px;
}
.tab-panel .tab-img {
  flex: 1;
  min-width: 260px;
  display: flex;
  justify-content: center;
  align-items: center;
}
.tab-panel .tab-img img {
  width: 100%;
  max-width: 340px;
  border-radius: 16px;
  box-shadow: 0 4px 24px rgba(0,255,231,0.10);
  background: #12263a;
}
.main-content {
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}
</style>

<!-- 场景和案例区块 -->
<style>
.scenario-case-section {
  margin: 64px 0 48px 0;
  background: #0f1c3a;
  border-radius: 18px;
  padding: 32px 0 32px 0;
  color: #fff;
}
.scenario-case-title {
  font-size: 2em;
  font-weight: bold;
  text-align: center;
  margin-bottom: 24px;
}
.scenario-case-main {
  display: flex;
  flex-wrap: wrap;
  max-width: 1200px;
  margin: 0 auto;
  min-height: 340px;
}
.scenario-tabs {
  flex: 0 0 160px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  justify-content: center;
  align-items: center;
  z-index: 2;
}
.scenario-tab {
  width: 120px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: rgba(255,255,255,0.06);
  color: #c8d8f8;
  font-size: 1.1em;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.3s, color 0.3s;
  margin-bottom: 8px;
  position: relative;
}
.scenario-tab.active {
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  color: #003a5d;
  font-weight: bold;
}
.scenario-tab .icon {
  width: 28px;
  height: 28px;
  margin-right: 8px;
}
.scenario-case-content {
  flex: 1;
  min-width: 0;
  position: relative;
  display: flex;
  align-items: stretch;
  overflow: hidden;
}
.scenario-case-bg {
  position: absolute;
  left: 0; top: 0; right: 0; bottom: 0;
  width: 100%; height: 100%;
  object-fit: cover;
  opacity: 0.22;
  z-index: 0;
}
.scenario-case-detail {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 700px;
  padding: 32px 40px 32px 40px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: flex-start;
}
.scenario-case-detail .case-title {
  display: flex;
  align-items: center;
  font-size: 1.2em;
  font-weight: bold;
  margin-bottom: 18px;
}
.scenario-case-detail .case-title .icon {
  width: 32px;
  height: 32px;
  margin-right: 10px;
}
.scenario-case-detail .case-headline {
  font-size: 1.5em;
  font-weight: bold;
  margin-bottom: 12px;
}
.scenario-case-detail .case-desc {
  font-size: 1.05em;
  color: #e0eaff;
  margin-bottom: 32px;
  line-height: 1.8;
}
.scenario-case-detail .case-btn {
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  color: #003a5d;
  font-weight: bold;
  padding: 14px 48px;
  font-size: 1.1em;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  margin-top: 8px;
}
.scenario-case-icons {
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  gap: 48px;
  position: absolute;
  right: 40px;
  bottom: 32px;
  z-index: 2;
}
.scenario-case-icons .icon-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  opacity: 0.7;
}
.scenario-case-icons .icon-col.active {
  opacity: 1;
}
.scenario-case-icons .icon-col .icon {
  width: 38px;
  height: 38px;
  margin-bottom: 6px;
}
.scenario-case-icons .icon-col span {
  color: #e0eaff;
  font-size: 1em;
}
@media (max-width: 900px) {
  .scenario-case-main { flex-direction: column; }
  .scenario-tabs { flex-direction: row; flex: none; margin-bottom: 18px; }
  .scenario-tab { margin-bottom: 0; margin-right: 8px; }
  .scenario-case-content { min-height: 320px; }
  .scenario-case-detail { padding: 24px 12px 24px 12px; }
  .scenario-case-icons { right: 12px; bottom: 12px; gap: 18px; }
}
</style>

<!-- 脚本必须放置到index.html中 <script>
function showTab(idx) {
  document.getElementById('tab-btn-1').classList.remove('active');
  document.getElementById('tab-btn-2').classList.remove('active');
  document.getElementById('tab-panel-1').classList.remove('active');
  document.getElementById('tab-panel-2').classList.remove('active');
  document.getElementById('tab-btn-' + idx).classList.add('active');
  document.getElementById('tab-panel-' + idx).classList.add('active');
}
</script> -->
<!-- 大语言模型LLM咨询区块 -->
<div style="background: linear-gradient(135deg, #0f223a 0%, #1a3a5d 100%); padding: 48px 0 32px 0; border-radius: 0 0 32px 32px; text-align: center; color: #fff;">
  <div class="main-content">
    <h1 style="font-size: 2.8em; font-weight: bold; margin-bottom: 0.3em; background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI & 大语言模型(LLM)咨询服务</h1>
    <div style="font-size: 1.3em; margin-bottom: 1.5em; color: #b6eaff;">充分利用数据和人工智能，加速企业变革</div>
    <a href="#contact" style="text-decoration: none;">
      <button style="background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%); color: #003a5d; font-weight: bold; padding: 16px 48px; font-size: 1.2em; border: none; border-radius: 8px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,255,231,0.15);">立即咨询</button>
    </a>
    <!-- 大语言模型咨询服务区块 -->
    <div>
      <div class="tab-switcher main-content">
        <button class="tab-btn active" id="tab-btn-1" onmouseover="showTab(1)">大语言模型咨询服务</button>
        <button class="tab-btn" id="tab-btn-2" onmouseover="showTab(2)">企业级AI机器人研发</button>
      </div>
      <div class="tab-content-area main-content">
        <div class="tab-panel active" id="tab-panel-1">
          <div class="tab-info">
            <h3><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg"/>技术评估与选择</h3>
            <p>针对客户需求，评估大语言模型的适用性，对比各种AI模型（如ChatGPT、LLaMA、Alpaca等），为您推荐最佳的AI技术和工具，以达到预期效果。</p>
            <h3><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f916.svg"/>ChatGPT培训服务</h3>
            <p>通过专业的教学与实践操作，助您全面掌握ChatGPT相关技术及应用，确保将所学知识有效地整合到现有系统中，提供流畅的用户体验和无缝集成。</p>
          </div>
          <div class="tab-img">
            <img src="images/corps/llm_transformer.png" alt="技术评估与选择示意图"/>
          </div>
        </div>
        <div class="tab-panel" id="tab-panel-2">
          <div class="tab-info">
            <h3><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4aa.svg"/>训练与微调</h3>
            <p>对ChatGPT模型及其他大语言模型进行训练和微调，以便更好地理解客户特定领域的需求和知识，从而提升性能和准确度。</p>
            <h3><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f916.svg"/>企业级机器人构建与集成</h3>
            <p>将开源大语言模型与内部业务数据库结合，助您打造专属的ChatGPT。在确保数据安全的前提下，提高数据利用效率和成本效益。</p>
          </div>
          <div class="tab-img">
            <img src="images/corps/llm_rag.png" alt="企业级机器人研发示意图"/>
          </div>
        </div>
      </div>
    </div>
    <!-- 服务链条区块 -->
    <div style="margin: 64px 0 48px 0; text-align: center; color: #fff;" class="main-content">
      <div style="font-size:2.2em; font-weight:bold; margin-bottom: 18px;">服务链条</div>
      <div style="font-size:1.1em; max-width:900px; margin:0 auto 38px auto; color:#b6c8e6;">若您正寻求利用AI和大语言模型的力量来推进企业数字化进程，熵策将为您值得信赖的数字化转型伙伴。凭借丰富的大语言模型经验，我们为您提供量身打造的解决方案。</div>
      <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:24px; max-width:1200px; margin:0 auto 32px auto;">
        <div style="flex:1; min-width:200px; max-width:240px; background:#11204a; border-radius:18px; padding:32px 18px 24px 18px; margin:0 4px; position:relative;">
        <div style="font-size:1.15em; font-weight:bold; margin-bottom:10px;">需求发现与评估</div>
        <div style="font-size:1em; color:#b6c8e6; margin-bottom:18px;">深入了解您的业务需求、目标及挑战，确定人工智能能为您带来价值的领域。</div>
        <div style="position:absolute; right:18px; bottom:18px; font-size:3.2em; color:#1a2c5d; font-weight:bold; opacity:0.25;">01</div>
        </div>
        <div style="flex:1; min-width:200px; max-width:240px; background:#11204a; border-radius:18px; padding:32px 18px 24px 18px; margin:0 4px; position:relative;">
        <div style="font-size:1.15em; font-weight:bold; margin-bottom:10px;">战略规划与实施路线图</div>
        <div style="font-size:1em; color:#b6c8e6; margin-bottom:18px;">制定个性化的AI战略及实施蓝图，确保与您的目标一致且满足特定需求。</div>
        <div style="position:absolute; right:18px; bottom:18px; font-size:3.2em; color:#1a2c5d; font-weight:bold; opacity:0.25;">02</div>
        </div>
        <div style="flex:1; min-width:200px; max-width:240px; background:#11204a; border-radius:18px; padding:32px 18px 24px 18px; margin:0 4px; position:relative;">
        <div style="font-size:1.15em; font-weight:bold; margin-bottom:10px;">解决方案设计与开发</div>
        <div style="font-size:1em; color:#b6c8e6; margin-bottom:18px;">运用适宜的大语言模型技术，为您设计并构建定制化AI解决方案，如企业专属聊天机器人。</div>
        <div style="position:absolute; right:18px; bottom:18px; font-size:3.2em; color:#1a2c5d; font-weight:bold; opacity:0.25;">03</div>
        </div>
        <div style="flex:1; min-width:200px; max-width:240px; background:#11204a; border-radius:18px; padding:32px 18px 24px 18px; margin:0 4px; position:relative;">
        <div style="font-size:1.15em; font-weight:bold; margin-bottom:10px;">部署整合与无缝集成</div>
        <div style="font-size:1em; color:#b6c8e6; margin-bottom:18px;">在现有系统和流程中顺利部署并集成AI解决方案，确保稳定运行。</div>
        <div style="position:absolute; right:18px; bottom:18px; font-size:3.2em; color:#1a2c5d; font-weight:bold; opacity:0.25;">04</div>
        </div>
        <div style="flex:1; min-width:200px; max-width:240px; background:#11204a; border-radius:18px; padding:32px 18px 24px 18px; margin:0 4px; position:relative;">
        <div style="font-size:1.15em; font-weight:bold; margin-bottom:10px;">持续监控、优化与支持</div>
        <div style="font-size:1em; color:#b6c8e6; margin-bottom:18px;">提供持续的监控、性能优化及支持服务，确保您的AI项目取得成功。</div>
        <div style="position:absolute; right:18px; bottom:18px; font-size:3.2em; color:#1a2c5d; font-weight:bold; opacity:0.25;">05</div>
        </div>
    </div>
    <a href="#learn-more" style="text-decoration: none;">
        <button style="background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%); color: #003a5d; font-weight: bold; padding: 14px 48px; font-size: 1.1em; border: none; border-radius: 8px; cursor: pointer; margin-top: 18px;">了解更多</button>
    </a>
    </div>
    <!-- 场景和案例区块-->
    <div class="scenario-case-section main-content">
    <div class="scenario-case-title">场景和案例</div>
    <div class="scenario-case-main">
        <div class="scenario-tabs">
        <div class="scenario-tab active" id="scenario-tab-0" onclick="showScenario(0)"><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6d2.svg"/>电子商务</div>
        <div class="scenario-tab" id="scenario-tab-1" onclick="showScenario(1)"><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ac.svg"/>数字营销</div>
        <div class="scenario-tab" id="scenario-tab-2" onclick="showScenario(2)"><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f465.svg"/>HR流程</div>
        <div class="scenario-tab" id="scenario-tab-3" onclick="showScenario(3)"><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c4.svg"/>项目管理</div>
        </div>
        <div class="scenario-case-content">
        <img class="scenario-case-bg" id="scenario-bg" src="images/corps/project-manage.png" alt="场景背景"/>
        <div class="scenario-case-detail" id="scenario-detail">
            <div class="case-title"><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6d2.svg"/>电子商务</div>
            <div class="case-headline">提升电商客户支持水平</div>
            <div class="case-desc">通过自动化客户支持流程，大语言模型助力电商企业优化客户体验。借助AI支持的对话功能，它能迅速、精准、高效地应对客户咨询，为企业节省客户支持相关的时间与成本，同时为客户提供高效且个性化的服务体验。此外，大语言模型还能协助电商企业洞察客户行为。通过分析客户对话数据，企业能更好地掌握客户偏好与需求，从而针对性地调整产品与服务。</div>
            <button class="case-btn">了解更多</button>
        </div>
        <div class="scenario-case-icons">
            <div class="icon-col active"><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6d2.svg"/><span>电子商务</span></div>
            <div class="icon-col"><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ac.svg"/><span>数字营销</span></div>
            <div class="icon-col"><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f465.svg"/><span>HR流程</span></div>
            <div class="icon-col"><img class="icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c4.svg"/><span>项目管理</span></div>
        </div>
        </div>
    </div>
</div>

<!-- 机器学习区块 -->
<style>
.ml-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.ml-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 12px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.ml-desc {
  font-size: 1.1em;
  color: #4a6a8a;
  margin-bottom: 24px;
}
.ml-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.ml-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.ml-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.ml-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
.ml-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .ml-wall { gap: 16px; }
  .ml-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .ml-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="ml-section main-content">
  <div class="ml-title">机器学习</div>
  <div class="ml-desc">机器学习是人工智能的核心子领域，专注于开发能够从数据中学习和改进的算法和统计模型。通过对大量数据的分析和学习，机器学习系统能够在没有明确编程的情况下执行特定任务。</div>
  <div class="ml-wall">
    <a class="ml-item" href="/#/ml/README" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="技术概览"/>
      <div class="ml-label">技术概览</div>
    </a>
    <a class="ml-item" href="/#/ml/02_关于机器学习_about_machine_learning" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e0.svg" alt="基础与范式"/>
      <div class="ml-label">基础与范式</div>
    </a>
    <a class="ml-item" href="/#/ml/03_监督学习_概要_监督学习_概要" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="监督学习"/>
      <div class="ml-label">监督学习</div>
    </a>
    <a class="ml-item" href="/#/ml/04_分类任务的处理流程_分类任务的处理流程" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="分类流程"/>
      <div class="ml-label">分类任务流程</div>
    </a>
    <a class="ml-item" href="/#/ml/05_分类_KNN_分类_KNN" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="KNN"/>
      <div class="ml-label">KNN算法</div>
    </a>
    <a class="ml-item" href="/#/ml/10_聚类-K-Means-找寻TopN关键词_聚类-K-Means-找寻TopN关键词" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="聚类"/>
      <div class="ml-label">K-Means聚类</div>
    </a>
    <a class="ml-item" href="/#/ml/15_多元线性回归预测转化率_多元线性回归预测转化率" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="回归"/>
      <div class="ml-label">多元线性回归</div>
    </a>
    <a class="ml-item" href="/#/ml/协同过滤推荐_协同过滤推荐" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="推荐系统"/>
      <div class="ml-label">协同过滤推荐</div>
    </a>
    <a class="ml-item" href="/#/ml/推荐中的冷启动问题_推荐中的冷启动问题" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="冷启动"/>
      <div class="ml-label">推荐冷启动问题</div>
    </a>
    <a class="ml-item" href="/#/ml/文本语义理解_文本语义理解" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ac.svg" alt="文本语义"/>
      <div class="ml-label">文本语义理解</div>
    </a>
    <a class="ml-item" href="/#/ml/EDA报告_EDA_Report" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ca.svg" alt="EDA"/>
      <div class="ml-label">EDA报告</div>
    </a>
    <a class="ml-item" href="/#/ml/21_标注工具" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f58c.svg" alt="标注工具"/>
      <div class="ml-label">标注工具</div>
    </a>
    <a class="ml-item" href="/#/ml/18_模型训练工具" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="训练工具"/>
      <div class="ml-label">模型训练工具</div>
    </a>
    <a class="ml-item" href="/#/ml/19_视觉图像处理数据集" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5bc.svg" alt="视觉数据集"/>
      <div class="ml-label">视觉图像数据集</div>
    </a>
    <a class="ml-item" href="/#/ml/20_yolo8_自定义数据集的训练" target="_blank">
      <img class="ml-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f916.svg" alt="YOLO8"/>
      <div class="ml-label">YOLO8自定义训练</div>
    </a>
  </div>
</div>

<!-- 机器学习数学与Python基础区块 -->
<style>
.mlbase-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.mlbase-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 12px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.mlbase-desc {
  font-size: 1.1em;
  color: #4a6a8a;
  margin-bottom: 24px;
}
.mlbase-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.mlbase-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: inherit;
}
.mlbase-item:hover {
  box-shadow: 0 6px 24px rgba(30,200,255,0.18);
  transform: translateY(-2px) scale(1.03);
}
.mlbase-icon {
  font-size: 2.2em;
  margin-bottom: 10px;
}
.mlbase-label {
  font-size: 1.08em;
  font-weight: 500;
  margin-bottom: 4px;
}
.mlbase-desc2 {
  font-size: 0.98em;
  color: #4a6a8a;
}
</style>

<div class="mlbase-section">
  <div class="mlbase-title">机器学习数学与Python基础</div>
  <div class="mlbase-desc">
    本区块系统梳理了机器学习所需的数学基础（线性代数、概率统计、微积分等）与Python数据科学工具（Numpy、Pandas等），为深入理解和实践AI/ML算法打下坚实基础。
  </div>
  <div class="mlbase-wall">
    <a class="mlbase-item" href="ml_base/01._基础语法.md">
      <div class="mlbase-icon">🐍</div>
      <div class="mlbase-label">Python基础语法</div>
      <div class="mlbase-desc2">Python基本语法与入门</div>
    </a>
    <a class="mlbase-item" href="ml_base/13._数据结构和推导式.md">
      <div class="mlbase-icon">📚</div>
      <div class="mlbase-label">数据结构与推导式</div>
      <div class="mlbase-desc2">列表、字典、集合等及推导式</div>
    </a>
    <a class="mlbase-item" href="ml_base/21._numpy.md">
      <div class="mlbase-icon">🔢</div>
      <div class="mlbase-label">Numpy基础</div>
      <div class="mlbase-desc2">数值计算核心库</div>
    </a>
    <a class="mlbase-item" href="ml_base/22._pandas基础.md">
      <div class="mlbase-icon">🗃️</div>
      <div class="mlbase-label">Pandas基础</div>
      <div class="mlbase-desc2">数据分析与处理</div>
    </a>
    <a class="mlbase-item" href="ml_base/38.1.线性代数.行列式.md">
      <div class="mlbase-icon">🧮</div>
      <div class="mlbase-label">线性代数基础</div>
      <div class="mlbase-desc2">行列式、矩阵、线性方程组等</div>
    </a>
    <a class="mlbase-item" href="ml_base/33._概率论和数理统计.md">
      <div class="mlbase-icon">🎲</div>
      <div class="mlbase-label">概率论与数理统计</div>
      <div class="mlbase-desc2">概率、分布、统计推断</div>
    </a>
    <a class="mlbase-item" href="ml_base/37._微积分.md">
      <div class="mlbase-icon">📈</div>
      <div class="mlbase-label">微积分基础</div>
      <div class="mlbase-desc2">导数、积分、极限等</div>
    </a>
    <a class="mlbase-item" href="ml_base/36.1_sympy-数学符号计算.md">
      <div class="mlbase-icon">🔣</div>
      <div class="mlbase-label">Sympy符号计算</div>
      <div class="mlbase-desc2">数学符号运算与表达式</div>
    </a>
    <a class="mlbase-item" href="ml_base/26._matplotlib数据可视化.md">
      <div class="mlbase-icon">📊</div>
      <div class="mlbase-label">Matplotlib数据可视化</div>
      <div class="mlbase-desc2">绘图与可视化基础</div>
    </a>
  </div>
</div>

<!-- 深度学习区块 -->
<style>
.dl-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.dl-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 12px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.dl-desc {
  font-size: 1.1em;
  color: #4a6a8a;
  margin-bottom: 24px;
}
.dl-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.dl-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: inherit;
}
.dl-item:hover {
  box-shadow: 0 6px 24px rgba(30,200,255,0.18);
  transform: translateY(-2px) scale(1.03);
}
.dl-icon {
  font-size: 2.2em;
  margin-bottom: 10px;
}
.dl-label {
  font-size: 1.08em;
  font-weight: 500;
  margin-bottom: 4px;
}
.dl-desc2 {
  font-size: 0.98em;
  color: #4a6a8a;
}
</style>

<div class="dl-section">
  <div class="dl-title">深度学习</div>
  <div class="dl-desc">
    本区块系统梳理了深度学习的核心理论、主流网络结构、训练技巧与工程实践，涵盖神经网络、卷积网络、循环网络、Transformer等内容。
  </div>
  <div class="dl-wall">
    <a class="dl-item" href="ml_deeplearning/README.md">
      <div class="dl-icon">🧠</div>
      <div class="dl-label">深度学习概览</div>
      <div class="dl-desc2">深度学习基础与发展</div>
    </a>
    <a class="dl-item" href="ml_deeplearning/01_神经网络基础.md">
      <div class="dl-icon">🔗</div>
      <div class="dl-label">神经网络基础</div>
      <div class="dl-desc2">感知机、前馈神经网络</div>
    </a>
    <a class="dl-item" href="ml_deeplearning/02_反向传播与优化.md">
      <div class="dl-icon">🔄</div>
      <div class="dl-label">反向传播与优化</div>
      <div class="dl-desc2">BP算法、梯度下降</div>
    </a>
    <a class="dl-item" href="ml_deeplearning/03_CNN卷积神经网络.md">
      <div class="dl-icon">🖼️</div>
      <div class="dl-label">卷积神经网络（CNN）</div>
      <div class="dl-desc2">图像处理与特征提取</div>
    </a>
    <a class="dl-item" href="ml_deeplearning/04_RNN循环神经网络.md">
      <div class="dl-icon">🔁</div>
      <div class="dl-label">循环神经网络（RNN）</div>
      <div class="dl-desc2">序列建模与时间序列</div>
    </a>
    <a class="dl-item" href="ml_deeplearning/05_Transformer与自注意力.md">
      <div class="dl-icon">⚡</div>
      <div class="dl-label">Transformer与自注意力</div>
      <div class="dl-desc2">NLP与大模型基础</div>
    </a>
    <a class="dl-item" href="ml_deeplearning/06_训练技巧与正则化.md">
      <div class="dl-icon">🛠️</div>
      <div class="dl-label">训练技巧与正则化</div>
      <div class="dl-desc2">Dropout、BatchNorm等</div>
    </a>
    <a class="dl-item" href="ml_deeplearning/07_深度学习工程实践.md">
      <div class="dl-icon">🏗️</div>
      <div class="dl-label">工程实践</div>
      <div class="dl-desc2">框架、部署与调优</div>
    </a>
  </div>
</div>

<!-- 机器学习经典算法区块 -->
<style>
.mlclassic-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.mlclassic-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 12px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.mlclassic-desc {
  font-size: 1.1em;
  color: #4a6a8a;
  margin-bottom: 24px;
}
.mlclassic-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.mlclassic-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: inherit;
}
.mlclassic-item:hover {
  box-shadow: 0 6px 24px rgba(30,200,255,0.18);
  transform: translateY(-2px) scale(1.03);
}
.mlclassic-icon {
  font-size: 2.2em;
  margin-bottom: 10px;
}
.mlclassic-label {
  font-size: 1.08em;
  font-weight: 500;
  margin-bottom: 4px;
}
.mlclassic-desc2 {
  font-size: 0.98em;
  color: #4a6a8a;
}
</style>

<div class="mlclassic-section">
  <div class="mlclassic-title">机器学习经典算法</div>
  <div class="mlclassic-desc">
    本区块系统梳理了机器学习领域的经典算法，包括回归、分类、聚类、降维等，涵盖理论基础与工程实践。
  </div>
  <div class="mlclassic-wall">
    <a class="mlclassic-item" href="ml_machinelearning/README.md">
      <div class="mlclassic-icon">📖</div>
      <div class="mlclassic-label">算法总览</div>
      <div class="mlclassic-desc2">机器学习经典算法概览</div>
    </a>
    <a class="mlclassic-item" href="ml_machinelearning/01_线性回归.md">
      <div class="mlclassic-icon">📈</div>
      <div class="mlclassic-label">线性回归</div>
      <div class="mlclassic-desc2">最基础的回归算法</div>
    </a>
    <a class="mlclassic-item" href="ml_machinelearning/02_逻辑回归.md">
      <div class="mlclassic-icon">🔢</div>
      <div class="mlclassic-label">逻辑回归</div>
      <div class="mlclassic-desc2">二分类与概率输出</div>
    </a>
    <a class="mlclassic-item" href="ml_machinelearning/03_KNN.md">
      <div class="mlclassic-icon">👥</div>
      <div class="mlclassic-label">K近邻（KNN）</div>
      <div class="mlclassic-desc2">基于距离的分类</div>
    </a>
    <a class="mlclassic-item" href="ml_machinelearning/04_决策树.md">
      <div class="mlclassic-icon">🌳</div>
      <div class="mlclassic-label">决策树</div>
      <div class="mlclassic-desc2">树结构的分类与回归</div>
    </a>
    <a class="mlclassic-item" href="ml_machinelearning/05_随机森林.md">
      <div class="mlclassic-icon">🌲</div>
      <div class="mlclassic-label">随机森林</div>
      <div class="mlclassic-desc2">集成学习代表算法</div>
    </a>
    <a class="mlclassic-item" href="ml_machinelearning/06_SVM支持向量机.md">
      <div class="mlclassic-icon">⚖️</div>
      <div class="mlclassic-label">支持向量机（SVM）</div>
      <div class="mlclassic-desc2">最大间隔分类器</div>
    </a>
    <a class="mlclassic-item" href="ml_machinelearning/07_聚类算法.md">
      <div class="mlclassic-icon">🔗</div>
      <div class="mlclassic-label">聚类算法</div>
      <div class="mlclassic-desc2">K-Means、层次聚类等</div>
    </a>
    <a class="mlclassic-item" href="ml_machinelearning/08_降维与特征选择.md">
      <div class="mlclassic-icon">📉</div>
      <div class="mlclassic-label">降维与特征选择</div>
      <div class="mlclassic-desc2">PCA、LDA等</div>
    </a>
    <a class="mlclassic-item" href="ml_machinelearning/09_集成学习.md">
      <div class="mlclassic-icon">🧩</div>
      <div class="mlclassic-label">集成学习</div>
      <div class="mlclassic-desc2">Bagging、Boosting等</div>
    </a>
  </div>
</div>

<!-- 大语言模型之词向量区块 -->
<style>
.word2vec-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.word2vec-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 12px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.word2vec-desc {
  font-size: 1.1em;
  color: #4a6a8a;
  margin-bottom: 24px;
}
.word2vec-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.word2vec-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: inherit;
}
.word2vec-item:hover {
  box-shadow: 0 6px 24px rgba(30,200,255,0.18);
  transform: translateY(-2px) scale(1.03);
}
.word2vec-icon {
  font-size: 2.2em;
  margin-bottom: 10px;
}
.word2vec-label {
  font-size: 1.08em;
  font-weight: 500;
  margin-bottom: 4px;
}
.word2vec-desc2 {
  font-size: 0.98em;
  color: #4a6a8a;
}
</style>

<div class="word2vec-section">
  <div class="word2vec-title">大语言模型之词向量</div>
  <div class="word2vec-desc">
    本区块系统梳理了词向量（Word Embedding）相关理论、主流模型与工程实践，涵盖Word2Vec、GloVe、FastText等内容。
  </div>
  <div class="word2vec-wall">
    <a class="word2vec-item" href="ml_word2vec/README.md">
      <div class="word2vec-icon">📝</div>
      <div class="word2vec-label">词向量概览</div>
      <div class="word2vec-desc2">词向量基础与发展</div>
    </a>
    <a class="word2vec-item" href="ml_word2vec/01_word2vec原理.md">
      <div class="word2vec-icon">🔤</div>
      <div class="word2vec-label">Word2Vec原理</div>
      <div class="word2vec-desc2">Skip-gram、CBOW等</div>
    </a>
    <a class="word2vec-item" href="ml_word2vec/02_GloVe原理.md">
      <div class="word2vec-icon">🧩</div>
      <div class="word2vec-label">GloVe原理</div>
      <div class="word2vec-desc2">全局向量建模</div>
    </a>
    <a class="word2vec-item" href="ml_word2vec/03_FastText原理.md">
      <div class="word2vec-icon">⚡</div>
      <div class="word2vec-label">FastText原理</div>
      <div class="word2vec-desc2">子词建模与高效训练</div>
    </a>
    <a class="word2vec-item" href="ml_word2vec/04_词向量可视化.md">
      <div class="word2vec-icon">📊</div>
      <div class="word2vec-label">词向量可视化</div>
      <div class="word2vec-desc2">降维与可视化方法</div>
    </a>
    <a class="word2vec-item" href="ml_word2vec/05_工程实践与应用.md">
      <div class="word2vec-icon">🛠️</div>
      <div class="word2vec-label">工程实践与应用</div>
      <div class="word2vec-desc2">实际项目中的词向量</div>
    </a>
  </div>
</div>

<!-- 大语言模型 LLM 知识 -->
<style>
.llm-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.llm-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.llm-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.llm-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.llm-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.llm-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
.llm-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .llm-wall { gap: 16px; }
  .llm-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .llm-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="llm-section main-content">
  <div class="llm-title">大语言模型 LLM 知识</div>
  <div class="llm-wall">
    <a class="llm-item" href="/#/llm/1.语言模型" target="_blank">
      <img class="llm-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="语言模型"/>
      <div class="llm-label">01_语言模型基础</div>
    </a>
    <a class="llm-item" href="/#/llm/02_词向量_word2vec" target="_blank">
      <img class="llm-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f521.svg" alt="词向量"/>
      <div class="llm-label">02_词向量与Word2Vec</div>
    </a>
    <a class="llm-item" href="/#/llm/NLP三大特征抽取器（CNN-RNN-TF）" target="_blank">
      <img class="llm-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9f2.svg" alt="特征抽取器"/>
      <div class="llm-label">03_NLP三大特征抽取器</div>
    </a>
    <a class="llm-item" href="/#/llm/NLP_核心的几个概念" target="_blank">
      <img class="llm-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e0.svg" alt="BERT与NLP"/>
      <div class="llm-label">04_BERT与NLP核心</div>
    </a>
    <a class="llm-item" href="/#/llm/01_transformer" target="_blank">
      <img class="llm-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f500.svg" alt="Transformer"/>
      <div class="llm-label">05_Transformer原理</div>
    </a>
    <a class="llm-item" href="/#/llm/1.llm概念" target="_blank">
      <img class="llm-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3d7.svg" alt="LLM架构"/>
      <div class="llm-label">06_LLM架构与主流模型</div>
    </a>
    <a class="llm-item" href="/#/llm/01_介绍_intro" target="_blank">
      <img class="llm-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e0.svg" alt="微调技术"/>
      <div class="llm-label">07_LLM微调技术</div>
    </a>
    <a class="llm-item" href="/#/llm/LLM为什么DecoderOnly架构" target="_blank">
      <img class="llm-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6a6.svg" alt="DecoderOnly"/>
      <div class="llm-label">08_DecoderOnly架构</div>
    </a>
    <a class="llm-item" href="/#/llm/README" target="_blank">
      <img class="llm-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f310.svg" alt="LLM知识总览"/>
      <div class="llm-label">09_LLM知识总览</div>
    </a>
  </div>
</div>

<style>
.llm-arch-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.llm-arch-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.llm-arch-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.llm-arch-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.llm-arch-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.llm-arch-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
.llm-arch-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .llm-arch-wall { gap: 16px; }
  .llm-arch-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .llm-arch-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="llm-arch-section main-content">
  <div class="llm-arch-title">LLM架构知识</div>
  <div class="llm-arch-wall">
    <a class="llm-arch-item" href="/#/llm_architecture/1.attention/1.attention" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="Attention"/>
      <div class="llm-arch-label">Attention机制</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/2.layer_normalization/2.layer_normalization" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ea.svg" alt="LayerNorm"/>
      <div class="llm-arch-label">Layer Normalization</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/3.位置编码/3.位置编码" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="位置编码"/>
      <div class="llm-arch-label">位置编码</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/4.tokenize分词/4.tokenize分词" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="分词"/>
      <div class="llm-arch-label">Tokenize分词</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/5.token及模型参数/5.token及模型参数" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ca.svg" alt="模型参数"/>
      <div class="llm-arch-label">Token及模型参数</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/6.激活函数/6.激活函数" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ec.svg" alt="激活函数"/>
      <div class="llm-arch-label">激活函数</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/MHA_MQA_GQA/MHA_MQA_GQA" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f500.svg" alt="MHA_MQA_GQA"/>
      <div class="llm-arch-label">MHA/MQA/GQA</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/解码策略（Top-k & Top-p & Temperatu/解码策略（Top-k & Top-p & Temperature）" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="解码策略"/>
      <div class="llm-arch-label">解码策略（Top-k/Top-p/Temperature）</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/bert细节/bert细节" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e0.svg" alt="BERT细节"/>
      <div class="llm-arch-label">BERT细节</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/bert变种/bert变种" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3a8.svg" alt="BERT变种"/>
      <div class="llm-arch-label">BERT变种</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/Transformer架构细节/Transformer架构细节" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="Transformer细节"/>
      <div class="llm-arch-label">Transformer架构细节</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/llama系列模型/llama系列模型" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f42a.svg" alt="llama系列"/>
      <div class="llm-arch-label">llama系列模型</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/llama 2代码详解/llama 2代码详解" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="llama2代码详解"/>
      <div class="llm-arch-label">llama 2代码详解</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/llama 3/llama 3" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3d7.svg" alt="llama3"/>
      <div class="llm-arch-label">llama 3</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/chatglm系列模型/chatglm系列模型" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f916.svg" alt="chatglm系列"/>
      <div class="llm-arch-label">chatglm系列模型</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/1.MoE论文/1.MoE论文" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f393.svg" alt="MoE论文"/>
      <div class="llm-arch-label">1.MoE论文</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/2.MoE经典论文简牍/2.MoE经典论文简牍" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="MoE经典论文"/>
      <div class="llm-arch-label">2.MoE经典论文简牍</div>
    </a>
    <a class="llm-arch-item" href="/#/llm_architecture/3.LLM MoE ：Switch Transformers/3.LLM MoE ：Switch Transformers" target="_blank">
      <img class="llm-arch-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f504.svg" alt="Switch Transformers"/>
      <div class="llm-arch-label">LLM MoE：Switch Transformers</div>
    </a>
  </div>
</div>

<style>
.llm-dataset-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.llm-dataset-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.llm-dataset-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.llm-dataset-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.llm-dataset-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.llm-dataset-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
.llm-dataset-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .llm-dataset-wall { gap: 16px; }
  .llm-dataset-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .llm-dataset-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="llm-dataset-section main-content">
  <div class="llm-dataset-title">LLM训练数据集</div>
  <div class="llm-dataset-wall">
    <a class="llm-dataset-item" href="/#/llm_training_datasets/README" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c2.svg" alt="训练数据集总览"/>
      <div class="llm-dataset-label">训练数据集总览</div>
    </a>
    <a class="llm-dataset-item" href="/#/llm_training_datasets/数据集" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">数据集格式与来源</div>
    </a>
  </div>
</div>

<div class="llm-dataset-section main-content">
  <div class="llm-dataset-title">LLM 微调指令集构建（基建）</div>
  <div class="llm-dataset-wall">
    <a class="llm-dataset-item" href="/#/llm_self_instruction/README" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">Self-Instruction应用大模型构建QA数据集</div>
    </a>
    <a class="llm-dataset-item" href="/#/llm_self_instruction/build_struction_instruction_movie" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">从结构化数据中构建QA指令集(电影)</div>
    </a>
    <a class="llm-dataset-item" href="/#/llm_self_instruction/build_struction_instruction_museum" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">从结构化数据中构建QA指令集(博物馆)</div>
    </a>
    <a class="llm-dataset-item" href="/#/llm_self_instruction/build_struction_instruction_museum_upgrade" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">从结构化数据中构建QA指令集(博物馆升级)</div>
    </a>
  </div>
</div>

<div class="llm-dataset-section main-content">
  <div class="llm-dataset-title">从零开始构建一个大模型</div>
  <div class="llm-dataset-wall">
    <a class="llm-dataset-item" href="/#/llm_create_model_by_self/README" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">构建一个弱弱的GPT-2的模型</div>
    </a>
    <a class="llm-dataset-item" href="/#/llm_create_model_by_self/resolve_1" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">更加正确的模型构建（一）</div>
    </a>
    <a class="llm-dataset-item" href="/#/llm_create_model_by_self/resolve_2" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">更加正确的模型构建（二）</div>
    </a>
    <a class="llm-dataset-item" href="/#/llm_create_model_by_self/从预训练到dpo_lora" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">基于 GPT-2 架构的因果语言模型（CLM）</div>
    </a>
  </div>
</div>

<div class="llm-dataset-section main-content">
  <div class="llm-dataset-title">角色扮演</div>
  <div class="llm-dataset-wall">
    <a class="llm-dataset-item" href="/#/llm_role_play/基于Baichuan2的角色扮演模型微调详细实现" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">基于Baichuan2的角色扮演模型微调详细实现</div>
    </a>
    <a class="llm-dataset-item" href="/#/llm_self_instruction/build_struction_instruction_museum_upgrade" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">从结构化数据中构建QA指令集(博物馆升级)</div>
    </a>
  </div>
</div>

<div class="llm-dataset-section main-content">
  <div class="llm-dataset-title">对话信息抽取</div>
  <div class="llm-dataset-wall">
    <a class="llm-dataset-item" href="/#/llm_dialog_element_extract/README" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">医疗信息对话信息抽取（Qwen模型微调）</div>
    </a>
    <a class="llm-dataset-item" href="/#/llm_self_instruction/build_struction_instruction_museum_upgrade" target="_blank">
      <img class="llm-dataset-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="数据集格式"/>
      <div class="llm-dataset-label">从结构化数据中构建QA指令集(博物馆升级)</div>
    </a>
  </div>
</div>

<style>
.llm-dist-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.llm-dist-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.llm-dist-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.llm-dist-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.llm-dist-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.llm-dist-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
.llm-dist-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .llm-dist-wall { gap: 16px; }
  .llm-dist-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .llm-dist-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="llm-dist-section main-content">
  <div class="llm-dist-title">LLM分布式训练</div>
  <div class="llm-dist-wall">
    <a class="llm-dist-item" href="/#/llm_distribute_training/README" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c2.svg" alt="分布式训练总览"/>
      <div class="llm-dist-label">分布式训练总览</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/1.概述/1.概述" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="概述"/>
      <div class="llm-dist-label">分布式训练概述</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/1.显存问题/1.显存问题" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5a5.svg" alt="显存问题"/>
      <div class="llm-dist-label">显存问题</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/2.数据并行/2.数据并行" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f465.svg" alt="数据并行"/>
      <div class="llm-dist-label">数据并行</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/3.流水线并行/3.流水线并行" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="流水线并行"/>
      <div class="llm-dist-label">流水线并行</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/4.张量并行/4.张量并行" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="张量并行"/>
      <div class="llm-dist-label">张量并行</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/5.序列并行/5.序列并行" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d1.svg" alt="序列并行"/>
      <div class="llm-dist-label">序列并行</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/6.多维度混合并行/6.多维度混合并行" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f52c.svg" alt="多维混合并行"/>
      <div class="llm-dist-label">多维度混合并行</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/7.自动并行/7.自动并行" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f916.svg" alt="自动并行"/>
      <div class="llm-dist-label">自动并行</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/8.moe并行/8.moe并行" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f393.svg" alt="moe并行"/>
      <div class="llm-dist-label">MoE并行</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/9.总结/9.总结" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4cb.svg" alt="总结"/>
      <div class="llm-dist-label">分布式训练总结</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/deepspeed介绍/deepspeed介绍" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f680.svg" alt="deepspeed"/>
      <div class="llm-dist-label">DeepSpeed介绍</div>
    </a>
    <a class="llm-dist-item" href="/#/llm_distribute_training/分布式训练题目/分布式训练题目" target="_blank">
      <img class="llm-dist-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2753.svg" alt="分布式训练题目"/>
      <div class="llm-dist-label">分布式训练题目</div>
    </a>
  </div>
</div>

<style>
.llm-sft-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.llm-sft-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.llm-sft-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.llm-sft-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.llm-sft-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.llm-sft-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
.llm-sft-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .llm-sft-wall { gap: 16px; }
  .llm-sft-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .llm-sft-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="llm-sft-section main-content">
  <div class="llm-sft-title">有监督微调</div>
  <div class="llm-sft-wall">
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/README" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c2.svg" alt="总览"/>
      <div class="llm-sft-label">有监督微调总览</div>
    </a>
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/1.基本概念/1.基本概念" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="基本概念"/>
      <div class="llm-sft-label">基本概念</div>
    </a>
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/1.微调/1.微调" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="微调"/>
      <div class="llm-sft-label">微调方法</div>
    </a>
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/2.预训练/2.预训练" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="预训练"/>
      <div class="llm-sft-label">预训练</div>
    </a>
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/2.prompting/2.prompting" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ac.svg" alt="prompting"/>
      <div class="llm-sft-label">Prompting</div>
    </a>
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/3.adapter-tuning/3.adapter-tuning" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ea.svg" alt="adapter-tuning"/>
      <div class="llm-sft-label">Adapter Tuning</div>
    </a>
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/4.lora/4.lora" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f52c.svg" alt="lora"/>
      <div class="llm-sft-label">LoRA</div>
    </a>
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/5.总结/5.总结" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4cb.svg" alt="总结"/>
      <div class="llm-sft-label">有监督微调总结</div>
    </a>
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/llama2微调/llama2微调" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f42a.svg" alt="llama2微调"/>
      <div class="llm-sft-label">Llama2微调</div>
    </a>
    <a class="llm-sft-item" href="/#/llm_supervised_fine_tuning/ChatGLM3微调/ChatGLM3微调" target="_blank">
      <img class="llm-sft-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f916.svg" alt="ChatGLM3微调"/>
      <div class="llm-sft-label">ChatGLM3微调</div>
    </a>
  </div>
</div>

<style>
.llm-infer-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.llm-infer-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.llm-infer-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.llm-infer-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.llm-infer-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.llm-infer-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
.llm-infer-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .llm-infer-wall { gap: 16px; }
  .llm-infer-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .llm-infer-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="llm-infer-section main-content">
  <div class="llm-infer-title">大语言模型推理</div>
  <div class="llm-infer-wall">
    <a class="llm-infer-item" href="/#/llm_inference/README" target="_blank">
      <img class="llm-infer-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c2.svg" alt="推理总览"/>
      <div class="llm-infer-label">推理总览</div>
    </a>
    <a class="llm-infer-item" href="/#/llm_inference/0.llm推理框架简单总结/0.llm推理框架简单总结" target="_blank">
      <img class="llm-infer-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="推理框架总结"/>
      <div class="llm-infer-label">推理框架总结</div>
    </a>
    <a class="llm-infer-item" href="/#/llm_inference/1.推理/1.推理" target="_blank">
      <img class="llm-infer-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f916.svg" alt="推理原理"/>
      <div class="llm-infer-label">推理原理</div>
    </a>
    <a class="llm-infer-item" href="/#/llm_inference/1.vllm/1.vllm" target="_blank">
      <img class="llm-infer-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f680.svg" alt="vllm"/>
      <div class="llm-infer-label">vLLM</div>
    </a>
    <a class="llm-infer-item" href="/#/llm_inference/2.text_generation_inference/2.text_generation_inference" target="_blank">
      <img class="llm-infer-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ac.svg" alt="text_generation_inference"/>
      <div class="llm-infer-label">Text Generation Inference</div>
    </a>
    <a class="llm-infer-item" href="/#/llm_inference/3.faster_transformer/3.faster_transformer" target="_blank">
      <img class="llm-infer-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26a1.svg" alt="faster_transformer"/>
      <div class="llm-infer-label">FasterTransformer</div>
    </a>
    <a class="llm-infer-item" href="/#/llm_inference/4.trt_llm/4.trt_llm" target="_blank">
      <img class="llm-infer-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="trt_llm"/>
      <div class="llm-infer-label">TRT-LLM</div>
    </a>
    <a class="llm-infer-item" href="/#/llm_inference/LLM推理常见参数/LLM推理常见参数" target="_blank">
      <img class="llm-infer-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="推理参数"/>
      <div class="llm-infer-label">推理常见参数</div>
    </a>
    <a class="llm-infer-item" href="/#/llm_inference/llm推理优化技术/llm推理优化技术" target="_blank">
      <img class="llm-infer-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ea.svg" alt="推理优化"/>
      <div class="llm-infer-label">推理优化技术</div>
    </a>
  </div>
</div>
<!-- 训练和模型压缩区块 -->
<style>
.training-compress-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.training-compress-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.training-compress-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.training-compress-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.training-compress-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.training-compress-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.training-compress-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
}
@media (max-width: 900px) {
  .training-compress-wall { gap: 16px; }
  .training-compress-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .training-compress-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="training-compress-section main-content">
  <div class="training-compress-title">训练和模型压缩</div>
  <div class="training-compress-wall">
    <a class="training-compress-item" href="/#/llm_training_and_compression/README" target="_blank">
      <img class="training-compress-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="高效微调"/>
      <div class="training-compress-label">高效微调与PEFT</div>
    </a>
    <a class="training-compress-item" href="/#/llm_training_and_compression/README" target="_blank">
      <img class="training-compress-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="模型压缩"/>
      <div class="training-compress-label">模型压缩技术</div>
    </a>
    <a class="training-compress-item" href="/#/llm_training_and_compression/5.高效训练&模型压缩" target="_blank">
      <img class="training-compress-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="分布式训练"/>
      <div class="training-compress-label">分布式训练与优化</div>
    </a>
    <a class="training-compress-item" href="/#/llm_training_and_compression/README" target="_blank">
      <img class="training-compress-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3c1.svg" alt="工业实践"/>
      <div class="training-compress-label">工业级实践与部署</div>
    </a>
    <a class="training-compress-item" href="/#/llm_training_and_compression/README" target="_blank">
      <img class="training-compress-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f52e.svg" alt="趋势展望"/>
      <div class="training-compress-label">趋势与前沿融合</div>
    </a>
    <a class="training-compress-item" href="/#/llm_training_and_compression/5.高效训练&模型压缩" target="_blank">
      <img class="training-compress-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="详细讲解"/>
      <div class="training-compress-label">详细讲解</div>
    </a>
  </div>
</div>
<!-- 提示词微调区块 -->
<style>
.prompt-tuning-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.prompt-tuning-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.prompt-tuning-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.prompt-tuning-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.prompt-tuning-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.prompt-tuning-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
.prompt-tuning-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .prompt-tuning-wall { gap: 16px; }
  .prompt-tuning-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .prompt-tuning-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="prompt-tuning-section main-content">
  <div class="prompt-tuning-title">提示词微调</div>
  <div class="prompt-tuning-wall">
    <a class="prompt-tuning-item" href="/#/llm_prompt_tuning/README" target="_blank">
      <img class="prompt-tuning-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="概述"/>
      <div class="prompt-tuning-label">Prompt Tuning 概述与背景</div>
    </a>
    <a class="prompt-tuning-item" href="/#/llm_prompt_tuning/4.PromptTuning" target="_blank">
      <img class="prompt-tuning-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="流程"/>
      <div class="prompt-tuning-label">Prompt-learning 流程与核心概念</div>
    </a>
    <a class="prompt-tuning-item" href="/#/llm_prompt_tuning/4.PromptTuning#主流模型适配" target="_blank">
      <img class="prompt-tuning-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="模型适配"/>
      <div class="prompt-tuning-label">主流模型适配</div>
    </a>
    <a class="prompt-tuning-item" href="/#/llm_prompt_tuning/4.PromptTuning#DeltaTuning" target="_blank">
      <img class="prompt-tuning-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="DeltaTuning"/>
      <div class="prompt-tuning-label">Delta Tuning/高效参数微调</div>
    </a>
    <a class="prompt-tuning-item" href="/#/llm_prompt_tuning/4.PromptTuning#趋势" target="_blank">
      <img class="prompt-tuning-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f680.svg" alt="趋势"/>
      <div class="prompt-tuning-label">工业实践与趋势</div>
    </a>
    <a class="prompt-tuning-item" href="/#/llm_prompt_tuning/4.PromptTuning" target="_blank">
      <img class="prompt-tuning-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="详细讲解"/>
      <div class="prompt-tuning-label">详细讲解</div>
    </a>
  </div>
</div>
<!-- 模型评估区块 -->
<style>
.eval-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.eval-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.eval-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.eval-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.eval-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.eval-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
.eval-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .eval-wall { gap: 16px; }
  .eval-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .eval-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="eval-section main-content">
  <div class="eval-title">模型评估</div>
  <div class="eval-wall">
    <a class="eval-item" href="/#/llm_evalution/README" target="_blank">
      <img class="eval-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ed.svg" alt="评估体系"/>
      <div class="eval-label">评估体系与核心维度</div>
    </a>
    <a class="eval-item" href="/#/llm_evalution/1.评测/1.评测" target="_blank">
      <img class="eval-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9d1-200d-2696-fe0f.svg" alt="评测方法"/>
      <div class="eval-label">自动与人工评测方法</div>
    </a>
    <a class="eval-item" href="/#/llm_evalution/1.大模型幻觉/1.大模型幻觉" target="_blank">
      <img class="eval-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f300.svg" alt="幻觉问题"/>
      <div class="eval-label">大模型幻觉问题</div>
    </a>
    <a class="eval-item" href="/#/llm_evalution/2.幻觉来源与缓解/2.幻觉来源与缓解" target="_blank">
      <img class="eval-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e1.svg" alt="幻觉缓解"/>
      <div class="eval-label">幻觉来源与缓解</div>
    </a>
    <a class="eval-item" href="/#/llm_evalution/README" target="_blank">
      <img class="eval-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e0.svg" alt="评测工具"/>
      <div class="eval-label">评测工具与最佳实践</div>
    </a>
    <a class="eval-item" href="/#/llm_evalution/README" target="_blank">
      <img class="eval-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="详细讲解"/>
      <div class="eval-label">详细讲解</div>
    </a>
  </div>
</div>

<!-- 大语言模型之RAG区块 -->
<style>
.rag-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.rag-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 12px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.rag-desc {
  font-size: 1.1em;
  color: #4a6a8a;
  margin-bottom: 24px;
}
.rag-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.rag-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: inherit;
}
.rag-item:hover {
  box-shadow: 0 6px 24px rgba(30,200,255,0.18);
  transform: translateY(-2px) scale(1.03);
}
.rag-icon {
  font-size: 2.2em;
  margin-bottom: 10px;
}
.rag-label {
  font-size: 1.08em;
  font-weight: 500;
  margin-bottom: 4px;
}
.rag-desc2 {
  font-size: 0.98em;
  color: #4a6a8a;
}
</style>

<div class="rag-section">
  <div class="rag-title">大语言模型之RAG</div>
  <div class="rag-desc">
    本区块系统梳理了RAG（Retrieval-Augmented Generation）相关理论、主流技术与工程实践，涵盖检索增强生成、向量数据库、知识检索等内容。
  </div>
  <div class="rag-wall">
    <a class="rag-item" href="llm_rag/README.md">
      <div class="rag-icon">📖</div>
      <div class="rag-label">RAG概览</div>
      <div class="rag-desc2">RAG基础与发展</div>
    </a>
    <a class="rag-item" href="llm_rag/01_RAG原理.md">
      <div class="rag-icon">🔍</div>
      <div class="rag-label">RAG原理</div>
      <div class="rag-desc2">检索增强生成机制</div>
    </a>
    <a class="rag-item" href="llm_rag/02_向量数据库.md">
      <div class="rag-icon">🗂️</div>
      <div class="rag-label">向量数据库</div>
      <div class="rag-desc2">知识存储与检索</div>
    </a>
    <a class="rag-item" href="llm_rag/03_知识检索.md">
      <div class="rag-icon">📚</div>
      <div class="rag-label">知识检索</div>
      <div class="rag-desc2">高效信息检索方法</div>
    </a>
    <a class="rag-item" href="llm_rag/04_RAG工程实践.md">
      <div class="rag-icon">🛠️</div>
      <div class="rag-label">RAG工程实践</div>
      <div class="rag-desc2">实际项目中的RAG应用</div>
    </a>
  </div>
</div>

<!-- 知识图谱区块 -->
<style>
.kg-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.kg-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.kg-main {
  display: flex;
  flex-wrap: wrap;
  max-width: 1200px;
  margin: 0 auto;
  min-height: 340px;
}
.kg-tabs {
  flex: 0 0 220px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  justify-content: flex-start;
  align-items: flex-start;
  z-index: 2;
  margin-top: 12px;
}
.kg-tab {
  width: 180px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  border-radius: 12px;
  background: #f2f6fa;
  color: #003a5d;
  font-size: 1.08em;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.3s, color 0.3s;
  margin-bottom: 8px;
  position: relative;
  border: 1px solid #e0eaff;
  padding-left: 18px;
  text-align: left;
}
.kg-tab.active {
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  color: #003a5d;
  font-weight: bold;
  border: 1px solid #00ffe7;
}
.kg-content {
  flex: 1;
  min-width: 0;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: flex-start;
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.03);
  margin-left: 24px;
  padding: 0 24px;
}
.kg-detail {
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: flex-start;
  background: #fff;
}
.kg-link {
  font-size: 1.2em;
  font-weight: bold;
  margin-bottom: 18px;
  margin-top: 18px;
}
.kg-images {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 24px;
}
.kg-img {
  /* max-width: 600px; */
  width: 100%;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  background: #f2f6fa;
  object-fit: contain;
}
@media (max-width: 900px) {
  .kg-main { flex-direction: column; }
  .kg-tabs { flex-direction: row; flex: none; margin-bottom: 18px; margin-top: 0; align-items: stretch; }
  .kg-tab { margin-bottom: 0; margin-right: 8px; width: auto; min-width: 100px; justify-content: center; padding-left: 0; }
  .kg-content { min-height: 320px; margin-left: 0; padding: 0 8px; }
  .kg-detail { padding: 0; max-width: 100%; }
  .kg-img { max-width: 100%; }
}
</style>
<div class="kg-section main-content">
  <div class="kg-title">知识图谱</div>
  <div class="kg-main">
    <div class="kg-tabs" id="kg-tabs"></div>
    <div class="kg-content" id="kg-content"></div>
  </div>
</div>


<!-- 架构设计区块 -->
<style>
.architecture-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.architecture-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.architecture-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.architecture-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.architecture-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.architecture-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.architecture-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
}
@media (max-width: 900px) {
  .architecture-wall { gap: 16px; }
  .architecture-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .architecture-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="architecture-section main-content">
  <div class="architecture-title">架构设计</div>
  <div class="architecture-wall">
        <a class="architecture-item" href="/#/architecture/00_架构目标_object_of_architecture" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3af.svg" alt="架构目标"/>
      <div class="architecture-label">00_架构目标_object_of_architecture</div>
    </a>
    <a class="architecture-item" href="/#/architecture/01_架构本质_the_essence_of_architecture" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e0.svg" alt="架构本质"/>
      <div class="architecture-label">01_架构本质_the_essence_of_architecture</div>
    </a>
    <a class="architecture-item" href="/#/architecture/01_偏好_preference" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="偏好"/>
      <div class="architecture-label">01_偏好_preference</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_01_不要设计_donot_design" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26d4.svg" alt="不要设计"/>
      <div class="architecture-label">02_01_不要过度设计_donot_design</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_02_DID_DID" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f194.svg" alt="DID"/>
      <div class="architecture-label">02_02_DID_DID</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_03_8_2_原则_principle" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4cf.svg" alt="8_2原则"/>
      <div class="architecture-label">02_03_8_2_原则_principle</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_04_DNS_dns" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f310.svg" alt="DNS"/>
      <div class="architecture-label">02_04_DNS_dns</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_05_更少对象_less_objects" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2796.svg" alt="更少对象"/>
      <div class="architecture-label">02_05_更少对象_less_objects</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_复杂性能操作系统_complicated_performance_os" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="复杂性能操作系统"/>
      <div class="architecture-label">02_复杂性能操作系统_complicated_performance_os</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_复杂性能集群_complicated_performance_cluster" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5a5.svg" alt="复杂性能集群"/>
      <div class="architecture-label">02_复杂性能集群_complicated_performance_cluster</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_复杂性能高可用_complicated_performance_high_available" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f501.svg" alt="复杂性能高可用"/>
      <div class="architecture-label">02_复杂性能高可用_complicated_performance_high_available</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_复杂性能可扩展性_complicated_performance_scalability" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="复杂性能可扩展性"/>
      <div class="architecture-label">02_复杂性能可扩展性_complicated_performance_scalability</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_复杂性能成本_complicated_performance_cost" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4b0.svg" alt="复杂性能成本"/>
      <div class="architecture-label">02_复杂性能成本_complicated_performance_cost</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_复杂性能安全_complicated_peformance_security" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f512.svg" alt="复杂性能安全"/>
      <div class="architecture-label">02_复杂性能安全_complicated_peformance_security</div>
    </a>
    <a class="architecture-item" href="/#/architecture/02_复杂性能规模_complicated_performance_guimo" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ca.svg" alt="复杂性能规模"/>
      <div class="architecture-label">02_复杂性能规模_complicated_performance_guimo</div>
    </a>
    <a class="architecture-item" href="/#/architecture/03_1_原则合适_principle_suitble" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2705.svg" alt="原则合适"/>
      <div class="architecture-label">03_1_原则合适_principle_suitble</div>
    </a>
    <a class="architecture-item" href="/#/architecture/03_2_原则简单_principle_simple" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f7e2.svg" alt="原则简单"/>
      <div class="architecture-label">03_2_原则简单_principle_simple</div>
    </a>
    <a class="architecture-item" href="/#/architecture/03_3_原则评估_principle_eveluation" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="原则评估"/>
      <div class="architecture-label">03_3_原则评估_principle_eveluation</div>
    </a>
    <a class="architecture-item" href="/#/architecture/03_4_原则案例_pinciple_cases" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="原则案例"/>
      <div class="architecture-label">03_4_原则案例_pinciple_cases</div>
    </a>
    <a class="architecture-item" href="/#/architecture/04_1_设计识别复杂情况步骤1_design_recognize_complicated_situation_step1" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="设计识别复杂情况1"/>
      <div class="architecture-label">04_1_设计识别复杂情况步骤1_design_recognize_complicated_situation_step1</div>
    </a>
    <a class="architecture-item" href="/#/architecture/04_1_设计识别复杂情况步骤2_design_recognize_complicated_situation_step2" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="设计识别复杂情况2"/>
      <div class="architecture-label">04_1_设计识别复杂情况步骤2_design_recognize_complicated_situation_step2</div>
    </a>
    <a class="architecture-item" href="/#/architecture/04_1_设计识别复杂情况步骤3_design_recognize_complicated_situation_step3" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e9.svg" alt="设计识别复杂情况3"/>
      <div class="architecture-label">04_1_设计识别复杂情况步骤3_design_recognize_complicated_situation_step3</div>
    </a>
    <a class="architecture-item" href="/#/architecture/04_1_2_设计识别复杂情况步骤3_design_recognize_complicated_situation_step3" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e9.svg" alt="设计识别复杂情况3-2"/>
      <div class="architecture-label">04_1_2_设计识别复杂情况步骤3_design_recognize_complicated_situation_step3</div>
    </a>
    <a class="architecture-item" href="/#/architecture/04_1_设计识别复杂情况步骤4_design_recognize_complicated_situation_step4" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e0.svg" alt="设计识别复杂情况4"/>
      <div class="architecture-label">04_1_设计识别复杂情况步骤4_design_recognize_complicated_situation_step4</div>
    </a>
    <a class="architecture-item" href="/#/architecture/04_1_设计细节步骤5_design_detail_step5" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="设计细节步骤5"/>
      <div class="architecture-label">04_1_设计细节步骤5_design_detail_step5</div>
    </a>
    <a class="architecture-item" href="/#/architecture/05_读写分离_read_write_seperator" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f500.svg" alt="读写分离"/>
      <div class="architecture-label">05_读写分离_read_write_seperator</div>
    </a>
    <a class="architecture-item" href="/#/architecture/07_NoSQL_nosql" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c4.svg" alt="NoSQL"/>
      <div class="architecture-label">07_NoSQL_nosql</div>
    </a>
    <a class="architecture-item" href="/#/architecture/08_Redis介绍_redis_intro" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ca.svg" alt="Redis介绍"/>
      <div class="architecture-label">08_Redis介绍_redis_intro</div>
    </a>
    <a class="architecture-item" href="/#/architecture/09_列式数据库_column_db" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ca.svg" alt="列式数据库"/>
      <div class="architecture-label">09_列式数据库_column_db</div>
    </a>
    <a class="architecture-item" href="/#/architecture/09_1_列式数据库系列_column_db_searials" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ca.svg" alt="列式数据库系列"/>
      <div class="architecture-label">09_1_列式数据库系列_column_db_searials</div>
    </a>
    <a class="architecture-item" href="/#/architecture/09_2_达梦数据库介绍_dameng_db_intro" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f1e8-1f1f3.svg" alt="达梦数据库介绍"/>
      <div class="architecture-label">09_2_达梦数据库介绍_dameng_db_intro</div>
    </a>
    <a class="architecture-item" href="/#/architecture/09_3_列式数据库案例ClickHouse_column_db_cases_clickhouse" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f40e.svg" alt="ClickHouse案例"/>
      <div class="architecture-label">09_3_列式数据库案例ClickHouse_column_db_cases_clickhouse</div>
    </a>
    <a class="architecture-item" href="/#/architecture/10_文档数据库_document_db" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c4.svg" alt="文档数据库"/>
      <div class="architecture-label">10_文档数据库_document_db</div>
    </a>
    <a class="architecture-item" href="/#/architecture/10_文档数据库和Elasticsearch_document_db_and_elasticsearch" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="文档数据库和Elasticsearch"/>
      <div class="architecture-label">10_文档数据库和Elasticsearch_document_db_and_elasticsearch</div>
    </a>
    <a class="architecture-item" href="/#/architecture/11_Elasticsearch同义词_elasticsearch_synonyme" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f524.svg" alt="Elasticsearch同义词"/>
      <div class="architecture-label">11_Elasticsearch同义词_elasticsearch_synonyme</div>
    </a>
    <a class="architecture-item" href="/#/architecture/12_缓存架构_cache_architecture" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c3.svg" alt="缓存架构"/>
      <div class="architecture-label">12_缓存架构_cache_architecture</div>
    </a>
    <a class="architecture-item" href="/#/architecture/13_CAP_cap" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2696.svg" alt="CAP"/>
      <div class="architecture-label">13_CAP_cap</div>
    </a>
    <a class="architecture-item" href="/#/architecture/13_CAP选择_cap_choice" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3b2.svg" alt="CAP选择"/>
      <div class="architecture-label">13_CAP选择_cap_choice</div>
    </a>
    <a class="architecture-item" href="/#/architecture/14_敏感词_sensitive_words" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26d4.svg" alt="敏感词"/>
      <div class="architecture-label">14_敏感词_sensitive_words</div>
    </a>
    <a class="architecture-item" href="/#/architecture/15_多直播_multi_living" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4fa.svg" alt="多直播"/>
      <div class="architecture-label">15_多直播_multi_living</div>
    </a>
    <a class="architecture-item" href="/#/architecture/15_多直播v2_multi_living_v2" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4fa.svg" alt="多直播v2"/>
      <div class="architecture-label">15_多直播v2_multi_living_v2</div>
    </a>
    <a class="architecture-item" href="/#/architecture/16_分布式文件系统_distribute_filesystem" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c2.svg" alt="分布式文件系统"/>
      <div class="architecture-label">16_分布式文件系统_distribute_filesystem</div>
    </a>
    <a class="architecture-item" href="/#/architecture/17_接口保护_interface_protection" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e1.svg" alt="接口保护"/>
      <div class="architecture-label">17_接口保护_interface_protection</div>
    </a>
    <a class="architecture-item" href="/#/architecture/17_1_接口保护降级_interface_protection_cutdown" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2b07.svg" alt="接口保护降级"/>
      <div class="architecture-label">17_1_接口保护降级_interface_protection_cutdown</div>
    </a>
    <a class="architecture-item" href="/#/architecture/17_接口保护限制_interface_protection_limit" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26d4.svg" alt="接口保护限制"/>
      <div class="architecture-label">17_接口保护限制_interface_protection_limit</div>
    </a>
    <a class="architecture-item" href="/#/architecture/17_接口保护熔断_interface_protection_rongduan" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a5.svg" alt="接口保护熔断"/>
      <div class="architecture-label">17_接口保护熔断_interface_protection_rongduan</div>
    </a>
    <a class="architecture-item" href="/#/architecture/17_接口保护熔断Flask演示_interface_protection_rongduan_flask_demo" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f525.svg" alt="接口保护熔断Flask演示"/>
      <div class="architecture-label">17_接口保护熔断Flask演示_interface_protection_rongduan_flask_demo</div>
    </a>
    <a class="architecture-item" href="/#/architecture/17_接口保护限制2_interface_protection_limit_2" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26d4.svg" alt="接口保护限制2"/>
      <div class="architecture-label">17_接口保护限制2_interface_protection_limit_2</div>
    </a>
    <a class="architecture-item" href="/#/architecture/17_接口保护限制3_interface_protection_limit_3" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26d4.svg" alt="接口保护限制3"/>
      <div class="architecture-label">17_接口保护限制3_interface_protection_limit_3</div>
    </a>
    <a class="architecture-item" href="/#/architecture/17_接口保护限制查询_interface_protection_limit_query" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="接口保护限制查询"/>
      <div class="architecture-label">17_接口保护限制查询_interface_protection_limit_query</div>
    </a>
    <a class="architecture-item" href="/#/architecture/18_可扩展性介绍_scalability_intro" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="可扩展性介绍"/>
      <div class="architecture-label">18_可扩展性介绍_scalability_intro</div>
    </a>
    <a class="architecture-item" href="/#/architecture/18_可扩展性不同拆分_scalibility_different_split" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2702.svg" alt="可扩展性不同拆分"/>
      <div class="architecture-label">18_可扩展性不同拆分_scalibility_different_split</div>
    </a>
    <a class="architecture-item" href="/#/architecture/18_可扩展性3种方法_scalibility_3_methods" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="可扩展性3种方法"/>
      <div class="architecture-label">18_可扩展性3种方法_scalibility_3_methods</div>
    </a>
    <a class="architecture-item" href="/#/architecture/19_API网关OpenAPI_api_gateway_openapi" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f309.svg" alt="API网关OpenAPI"/>
      <div class="architecture-label">19_API网关OpenAPI_api_gateway_openapi</div>
    </a>
    <a class="architecture-item" href="/#/architecture/20_用户SSO认证_user_sso_auth" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f511.svg" alt="用户SSO认证"/>
      <div class="architecture-label">20_用户SSO认证_user_sso_auth</div>
    </a>
    <a class="architecture-item" href="/#/architecture/21_SSO演示_sso_demo" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3ac.svg" alt="SSO演示"/>
      <div class="architecture-label">21_SSO演示_sso_demo</div>
    </a>
    <a class="architecture-item" href="/#/architecture/22_高可用存储架构_ha_storage_architecture" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="高可用存储架构"/>
      <div class="architecture-label">22_高可用存储架构_ha_storage_architecture</div>
    </a>
    <a class="architecture-item" href="/#/architecture/README" target="_blank">
      <img class="architecture-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="README"/>
      <div class="architecture-label">README</div>
    </a>
  </div>
</div>



<!-- 架构设计关联知识区块 -->
<style>
.basic-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.basic-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.basic-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.basic-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.basic-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.basic-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.basic-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
}
@media (max-width: 900px) {
  .basic-wall { gap: 16px; }
  .basic-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .basic-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="basic-section main-content">
  <div class="basic-title">架构设计关联知识</div>
  <div class="basic-wall">
    <a class="basic-item" href="/#/basic/01_TCP三次握手_tcp_three_handshake" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f310.svg" alt="网络"/>
      <div class="basic-label">01_TCP三次握手_tcp_three_handshake</div>
    </a>
    <a class="basic-item" href="/#/basic/01_2_TCP三次握手_tcp_three_handshake" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="链接"/>
      <div class="basic-label">01_2_TCP三次握手_tcp_three_handshake</div>
    </a>
    <a class="basic-item" href="/#/basic/01_3_TCP防御SYN洪水攻击_tcp_defend_syn_flood" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e1.svg" alt="防御"/>
      <div class="basic-label">01_3_TCP防御SYN洪水攻击_tcp_defend_syn_flood</div>
    </a>
    <a class="basic-item" href="/#/basic/01_4_hping3测试SYN洪水攻击_hping3_test_synflood" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="测试"/>
      <div class="basic-label">01_4_hping3测试SYN洪水攻击_hping3_test_synflood</div>
    </a>
    <a class="basic-item" href="/#/basic/01_5_hping3测试SYN洪水攻击_hping3_test_synflood" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="测试"/>
      <div class="basic-label">01_5_hping3测试SYN洪水攻击_hping3_test_synflood</div>
    </a>
    <a class="basic-item" href="/#/basic/01_6_hping3测试SYN洪水攻击Docker_hping3_test_synflood_docker" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5a5.svg" alt="Docker"/>
      <div class="basic-label">01_6_hping3测试SYN洪水攻击Docker_hping3_test_synflood_docker</div>
    </a>
    <a class="basic-item" href="/#/basic/01_7_iOS_TCP_UDP_ios_tcp_udp" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4f1.svg" alt="iOS"/>
      <div class="basic-label">01_7_iOS_TCP_UDP_ios_tcp_udp</div>
    </a>
    <a class="basic-item" href="/#/basic/01_8_HTTP_http" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="HTTP"/>
      <div class="basic-label">01_8_HTTP_http</div>
    </a>
    <a class="basic-item" href="/#/basic/02_处理器线程_processor_thread" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9f5.svg" alt="处理器线程"/>
      <div class="basic-label">02_处理器线程_processor_thread</div>
    </a>
    <a class="basic-item" href="/#/basic/03_操作系统内存_os_memory" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="操作系统内存"/>
      <div class="basic-label">03_操作系统内存_os_memory</div>
    </a>
    <a class="basic-item" href="/#/basic/04_Java结构_java_structure" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9f1.svg" alt="Java结构"/>
      <div class="basic-label">04_Java结构_java_structure</div>
    </a>
    <a class="basic-item" href="/#/basic/05_模式_patterns" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3a8.svg" alt="模式"/>
      <div class="basic-label">05_模式_patterns</div>
    </a>
    <a class="basic-item" href="/#/basic/06_JVM_jvm" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2615.svg" alt="JVM"/>
      <div class="basic-label">06_JVM_jvm</div>
    </a>
    <a class="basic-item" href="/#/basic/06_2_JVM性能_jvm_performance" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26a1.svg" alt="JVM性能"/>
      <div class="basic-label">06_2_JVM性能_jvm_performance</div>
    </a>
    <a class="basic-item" href="/#/basic/06_3_JVM_Arthas_jvm_arthas" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="JVM_Arthas"/>
      <div class="basic-label">06_3_JVM_Arthas_jvm_arthas</div>
    </a>
    <a class="basic-item" href="/#/basic/06_4_JVM和交换_jvm_and_swap" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f504.svg" alt="JVM和交换"/>
      <div class="basic-label">06_4_JVM和交换_jvm_and_swap</div>
    </a>
    <a class="basic-item" href="/#/basic/07_线上问题_online_problem" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f41e.svg" alt="线上问题"/>
      <div class="basic-label">07_线上问题_online_problem</div>
    </a>
    <a class="basic-item" href="/#/basic/07_2_线上问题_online_problem" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f41e.svg" alt="线上问题2"/>
      <div class="basic-label">07_2_线上问题_online_problem</div>
    </a>
    <a class="basic-item" href="/#/basic/07_3_线上问题_online_problem" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f41e.svg" alt="线上问题3"/>
      <div class="basic-label">07_3_线上问题_online_problem</div>
    </a>
    <a class="basic-item" href="/#/basic/07_4_线上问题交换_online_problem_swap" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f504.svg" alt="线上问题交换"/>
      <div class="basic-label">07_4_线上问题交换_online_problem_swap</div>
    </a>
    <a class="basic-item" href="/#/basic/07_4_线上问题模拟交换_online_problem_mock_swap" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ea.svg" alt="线上问题模拟交换"/>
      <div class="basic-label">07_4_线上问题模拟交换_online_problem_mock_swap</div>
    </a>
    <a class="basic-item" href="/#/basic/07_4_线上问题模拟交换副本_online_problem_mock_swap copy" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ea.svg" alt="线上问题模拟交换副本"/>
      <div class="basic-label">07_4_线上问题模拟交换副本_online_problem_mock_swap copy</div>
    </a>
    <a class="basic-item" href="/#/basic/07_5_线上问题查找_online_problem_find" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="线上问题查找"/>
      <div class="basic-label">07_5_线上问题查找_online_problem_find</div>
    </a>
    <a class="basic-item" href="/#/basic/07_6_线上问题交换查找命令_online_problem_swap_find_command" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="交换查找命令"/>
      <div class="basic-label">07_6_线上问题交换查找命令_online_problem_swap_find_command</div>
    </a>
    <a class="basic-item" href="/#/basic/07_7_脚本记录_script_record" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dc.svg" alt="脚本记录"/>
      <div class="basic-label">07_7_脚本记录_script_record</div>
    </a>
    <a class="basic-item" href="/#/basic/07_8_HashMap_hashmap" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/0023-20e3.svg" alt="HashMap"/>
      <div class="basic-label">07_8_HashMap_hashmap</div>
    </a>
    <a class="basic-item" href="/#/basic/07_JVM交换测试_jvm_swap_test" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ea.svg" alt="JVM交换测试"/>
      <div class="basic-label">07_JVM交换测试_jvm_swap_test</div>
    </a>
    <a class="basic-item" href="/#/basic/07_JVM模拟分配内存Java_jvm_mock_allocate_memory_java" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="JVM模拟分配内存Java"/>
      <div class="basic-label">07_JVM模拟分配内存Java_jvm_mock_allocate_memory_java</div>
    </a>
    <a class="basic-item" href="/#/basic/07_交换内存监控_swap_memory_monitoring" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="交换内存监控"/>
      <div class="basic-label">07_交换内存监控_swap_memory_monitoring</div>
    </a>
    <a class="basic-item" href="/#/basic/07_memory_eater.py" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f40d.svg" alt="memory_eater.py"/>
      <div class="basic-label">07_memory_eater.py</div>
    </a>
    <a class="basic-item" href="/#/basic/08_线程基础_thread_basic" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9f5.svg" alt="线程基础"/>
      <div class="basic-label">08_线程基础_thread_basic</div>
    </a>
    <a class="basic-item" href="/#/basic/08_线程执行器文件下载_threadexcutor_filedownload" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2b07.svg" alt="线程执行器文件下载"/>
      <div class="basic-label">08_线程执行器文件下载_threadexcutor_filedownload</div>
    </a>
    <a class="basic-item" href="/#/basic/09_可重入锁_reentrant_lock" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f512.svg" alt="可重入锁"/>
      <div class="basic-label">09_可重入锁_reentrant_lock</div>
    </a>
    <a class="basic-item" href="/#/basic/10_1_ThreadLocal_threadlocal" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="ThreadLocal1"/>
      <div class="basic-label">10_1_ThreadLocal_threadlocal</div>
    </a>
    <a class="basic-item" href="/#/basic/10_2_ThreadLocal_threadlocal" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="ThreadLocal2"/>
      <div class="basic-label">10_2_ThreadLocal_threadlocal</div>
    </a>
    <a class="basic-item" href="/#/basic/11_JMM_volatile_jmm_volatile" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26a1.svg" alt="JMM_volatile"/>
      <div class="basic-label">11_JMM_volatile_jmm_volatile</div>
    </a>
    <a class="basic-item" href="/#/basic/12_SpringBean循环依赖" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/267b.svg" alt="SpringBean循环依赖"/>
      <div class="basic-label">12_SpringBean循环依赖</div>
    </a>
    <a class="basic-item" href="/#/basic/13_JWT的正确使用" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f511.svg" alt="JWT的正确使用"/>
      <div class="basic-label">13_JWT的正确使用</div>
    </a>
    <a class="basic-item" href="/#/basic/14_游标分页" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d1.svg" alt="游标分页"/>
      <div class="basic-label">14_游标分页</div>
    </a>
    <a class="basic-item" href="/#/basic/15_rest_graphql" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="rest_graphql"/>
      <div class="basic-label">15_rest_graphql</div>
    </a>
    <a class="basic-item" href="/#/basic/README" target="_blank">
      <img class="basic-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="README"/>
      <div class="basic-label">README</div>
    </a>
  </div>
</div>

<!-- 微服务建设区块 -->
<style>
.microservice-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.microservice-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.microservice-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.microservice-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.microservice-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.microservice-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
}
.microservice-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .microservice-wall { gap: 16px; }
  .microservice-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .microservice-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="microservice-section main-content">
  <div class="microservice-title">微服务建设</div>
  <div class="microservice-wall">
    <a class="microservice-item" href="/#/micro_service/01_关于一些原则_about_some_principles" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="原则"/>
      <div class="microservice-label">01 关于一些原则 about some principles</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/01_关于三层架构_about_three_level_architecture" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="三层架构"/>
      <div class="microservice-label">01 关于三层架构 about three level architecture</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/01_关于单体架构_about_monolithic_architecture" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="单体架构"/>
      <div class="microservice-label">01 关于单体架构 about monolithic architecture</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_2_关于分布式系统组件_about_comp_of_distribution_system" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5a5.svg" alt="分布式组件"/>
      <div class="microservice-label">02 2 关于分布式系统组件 about comp of distribution system</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_2_关于服务依赖_about_dependent_of_service" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="服务依赖"/>
      <div class="microservice-label">02 2 关于服务依赖 about dependent of service</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_2_关于自动部署_about_auto_deployment" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f528.svg" alt="自动部署"/>
      <div class="microservice-label">02 2 关于自动部署 about auto deployment</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_2_关于运维成本_about_operational_cost" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4b0.svg" alt="运维成本"/>
      <div class="microservice-label">02 2 关于运维成本 about operational cost</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_关于SOA和微服务_about_soa_and_microservice" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="SOA微服务"/>
      <div class="microservice-label">02 关于SOA和微服务 about soa and microservice</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_关于SOA现在_about_soa_now" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="SOA"/>
      <div class="microservice-label">02 关于SOA现在 about soa now</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_关于SRP_about_SRP" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="SRP"/>
      <div class="microservice-label">02 关于SRP about SRP</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_关于处理器违规_about_processor_violation" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="处理器违规"/>
      <div class="microservice-label">02 关于处理器违规 about processor violation</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_关于微服务_about_micro_service" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="微服务"/>
      <div class="microservice-label">02 关于微服务 about micro service</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_关于服务本质_about_essence_of_service" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="服务本质"/>
      <div class="microservice-label">02 关于服务本质 about essence of service</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_关于虚拟容器Docker_about_virtual_container_docker" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e0.svg" alt="Docker"/>
      <div class="microservice-label">02 关于虚拟容器Docker about virtual container docker</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/02_关于部署独立性_about_deployment_independency" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="部署独立性"/>
      <div class="microservice-label">02 关于部署独立性 about deployment independency</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/03_关于代码检查_about_code_checking" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="代码检查"/>
      <div class="microservice-label">03 关于代码检查 about code checking</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/04_关于Dockerfile_about_dockerfile" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e0.svg" alt="Dockerfile"/>
      <div class="microservice-label">04 关于Dockerfile about dockerfile</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/04_关于构建私有Docker仓库_about_building_private_docker_respository" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e0.svg" alt="私有仓库"/>
      <div class="microservice-label">04 关于构建私有Docker仓库 about building private docker respository</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_about_log_collection" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4cb.svg" alt="日志收集"/>
      <div class="microservice-label">05 about log collection</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_关于ELK栈_about_elk_stack" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="ELK"/>
      <div class="microservice-label">05 关于ELK栈 about elk stack</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_关于ELK栈详细_about_elk_stack_detailed" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="ELK"/>
      <div class="microservice-label">05 关于ELK栈详细 about elk stack detailed</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_关于Fluentd_ES_Kibana_about_fluentd_es_kibana" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Fluentd"/>
      <div class="microservice-label">05 关于Fluentd ES Kibana about fluentd es kibana</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_关于Fluentd_ES_Kibana详细_about_fluentd_es_kibana_detailed" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Fluentd"/>
      <div class="microservice-label">05 关于Fluentd ES Kibana详细 about fluentd es kibana detailed</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_关于K8s_about_k8s" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f40a.svg" alt="K8s"/>
      <div class="microservice-label">05 关于K8s about k8s</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_关于Loki_Promtail_Grafana_about_loki_promtail_grafana" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Loki"/>
      <div class="microservice-label">05 关于Loki Promtail Grafana about loki promtail grafana</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_关于Loki_Promtail_Grafana_第2部分_about_loki_promtail_grafana_part2" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Loki"/>
      <div class="microservice-label">05 关于Loki Promtail Grafana 第2部分 about loki promtail grafana part2</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_关于Loki_Promtail_Grafana详细_about_loki_promtail_grafana_detailed" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Loki"/>
      <div class="microservice-label">05 关于Loki Promtail Grafana详细 about loki promtail grafana detailed</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/05_关于持续CI_about_continual_ci" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f504.svg" alt="持续CI"/>
      <div class="microservice-label">05 关于持续CI about continual ci</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/06_关于K8s集群_about_k8s_cluster" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f40a.svg" alt="K8s集群"/>
      <div class="microservice-label">06 关于K8s集群 about k8s cluster</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/07_关于远程调用_about_remote_invocation" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="远程调用"/>
      <div class="microservice-label">07 关于远程调用 about remote invocation</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/07_关于阿里Dubbo_about_alibaba_dubbo" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="Dubbo"/>
      <div class="microservice-label">07 关于阿里Dubbo about alibaba dubbo</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/08_关于Session_Cookie_JWT_about_session_cookie_jwt" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f511.svg" alt="Session JWT"/>
      <div class="microservice-label">08 关于Session Cookie JWT about session cookie jwt</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/09_关于任务队列_about_queue_task" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="任务队列"/>
      <div class="microservice-label">09 关于任务队列 about queue task</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/09_关于数据同步队列Canal_about_queue_data_sync_canal" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="同步队列"/>
      <div class="microservice-label">09 关于数据同步队列Canal about queue data sync canal</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/09_关于数据同步队列_about_queue_data_sync" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="同步队列"/>
      <div class="microservice-label">09 关于数据同步队列 about queue data sync</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/09_关于消息队列_about_message_queue" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="消息队列"/>
      <div class="microservice-label">09 关于消息队列 about message queue</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/09_关于缓存队列_about_queue_buffer" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="缓存队列"/>
      <div class="microservice-label">09 关于缓存队列 about queue buffer</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/09_关于请求队列_about_queue_request" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="请求队列"/>
      <div class="microservice-label">09 关于请求队列 about queue request</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/10_关于高并发原则_about_hight_concurrency_principle" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f680.svg" alt="高并发"/>
      <div class="microservice-label">10 关于高并发原则 about hight concurrency principle</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/11_1_关于缓存CDN_about_cache_cdn" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="CDN缓存"/>
      <div class="microservice-label">11 1 关于缓存CDN about cache cdn</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/11_1_关于缓存Nginx_about_cache_nginx" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="Nginx缓存"/>
      <div class="microservice-label">11 1 关于缓存Nginx about cache nginx</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/11_1_关于缓存Redis_about_cache_redis" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="Redis缓存"/>
      <div class="microservice-label">11 1 关于缓存Redis about cache redis</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/11_1_关于缓存代理_about_cache_agent" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="缓存代理"/>
      <div class="microservice-label">11 1 关于缓存代理 about cache agent</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/11_关于缓存_about_cache" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="缓存"/>
      <div class="microservice-label">11 关于缓存 about cache</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/11_关于缓存恢复_about_cache_recover" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="缓存恢复"/>
      <div class="microservice-label">11 关于缓存恢复 about cache recover</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/11_关于缓存模式_about_cache_pattern" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="缓存模式"/>
      <div class="microservice-label">11 关于缓存模式 about cache pattern</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/12_3_负载均衡Nginx_Consul_loadbalance_nginx_consul" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2699.svg" alt="负载均衡"/>
      <div class="microservice-label">12 3 负载均衡Nginx Consul loadbalance nginx consul</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/12_关于负载均衡Nginx_about_loadbalance_nginx" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2699.svg" alt="负载均衡"/>
      <div class="microservice-label">12 关于负载均衡Nginx about loadbalance nginx</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/12_关于负载均衡和代理_about_loadbalance_and_proxy" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2699.svg" alt="负载均衡"/>
      <div class="microservice-label">12 关于负载均衡和代理 about loadbalance and proxy</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/12_关于负载均衡选择_about_loadbalance_choice" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2699.svg" alt="负载均衡"/>
      <div class="microservice-label">12 关于负载均衡选择 about loadbalance choice</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/13_关于隔离_about_isolation" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f512.svg" alt="隔离"/>
      <div class="microservice-label">13 关于隔离 about isolation</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/14_1_关于限流器Redis_Lua_about_limiter_redis_lua" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6ab.svg" alt="限流器"/>
      <div class="microservice-label">14 1 关于限流器Redis Lua about limiter redis lua</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/14_关于限流器_about_limiter" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6ab.svg" alt="限流器"/>
      <div class="microservice-label">14 关于限流器 about limiter</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/15_关于节流_about_throttle" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6ab.svg" alt="节流"/>
      <div class="microservice-label">15 关于节流 about throttle</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/16_关于层级_about_level" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="层级"/>
      <div class="microservice-label">16 关于层级 about level</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/17_关于超时_about_timeout" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f1.svg" alt="超时"/>
      <div class="microservice-label">17 关于超时 about timeout</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/18_关于回滚_about_rollback" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f504.svg" alt="回滚"/>
      <div class="microservice-label">18 关于回滚 about rollback</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/19_关于性能测试_about_performance_test" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3c6.svg" alt="性能测试"/>
      <div class="microservice-label">19 关于性能测试 about performance test</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/20_关于系统性能调优和灾难恢复_about_system_perform_tuning_and_disaser_recovery" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3c6.svg" alt="性能调优"/>
      <div class="microservice-label">20 关于系统性能调优和灾难恢复 about system perform tuning and disaser recovery</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/21_关于线程池CountDownLatch_about_threadpool_countdownlatch" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f528.svg" alt="线程池"/>
      <div class="microservice-label">21 关于线程池CountDownLatch about threadpool countdownlatch</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/21_关于线程池指南_about_threadpool_guide" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f528.svg" alt="线程池"/>
      <div class="microservice-label">21 关于线程池指南 about threadpool guide</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/22_关于性能合并请求_about_performance_mergerequire" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3c6.svg" alt="性能合并"/>
      <div class="microservice-label">22 关于性能合并请求 about performance mergerequire</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/22_关于性能调用_about_performance_call" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3c6.svg" alt="性能调用"/>
      <div class="microservice-label">22 关于性能调用 about performance call</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/23_about_scaling_db_sharding-jdbc" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="扩展"/>
      <div class="microservice-label">23 about scaling db sharding Jdbc</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/23_关于扩展_about_scaling" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="扩展"/>
      <div class="microservice-label">23 关于扩展 about scaling</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/23_关于扩展分布式ID_about_scaling_distributed_id" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="扩展分布式ID"/>
      <div class="microservice-label">23 关于扩展分布式ID about scaling distributed id</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/23_关于扩展数据库_about_scaling_db" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="扩展数据库"/>
      <div class="microservice-label">23 关于扩展数据库 about scaling db</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/23_关于扩展数据库分片策略_about_scaling_db_sharding_strages" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="扩展分片"/>
      <div class="microservice-label">23 关于扩展数据库分片策略 about scaling db sharding strages</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/23_关于扩展数据库应用_about_scaling_db_app" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="扩展应用"/>
      <div class="microservice-label">23 关于扩展数据库应用 about scaling db app</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/23_关于扩展绑定积分路由器_about_scaling_bind_integeral_router" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="扩展路由"/>
      <div class="microservice-label">23 关于扩展绑定积分路由器 about scaling bind integeral router</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/24_关于数据差异_about_data_diff" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ca.svg" alt="数据差异"/>
      <div class="microservice-label">24 关于数据差异 about data diff</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/25_关于调度器XXJob_about_scheduler_xxjob" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f3.svg" alt="调度器"/>
      <div class="microservice-label">25 关于调度器XXJob about scheduler xxjob</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/25_关于调度器_about_scheduler" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f3.svg" alt="调度器"/>
      <div class="microservice-label">25 关于调度器 about scheduler</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_2_关于Elasticsearch场景_about_elasticsearch_scense" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch"/>
      <div class="microservice-label">26 2 关于Elasticsearch场景 about elasticsearch scense</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch_Docker_about_elasticsearch_docker" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch Docker"/>
      <div class="microservice-label">26 关于Elasticsearch Docker about elasticsearch docker</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch_HTTPS_about_elasticasearch_https" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch HTTPS"/>
      <div class="microservice-label">26 关于Elasticsearch HTTPS about elasticasearch https</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch_Logstash_SSL_about_elasticsearch_logstash_ssl" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Logstash SSL"/>
      <div class="microservice-label">26 关于Elasticsearch Logstash SSL about elasticsearch logstash ssl</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch_Logstash_about_elasticsearch_logstash" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Logstash"/>
      <div class="microservice-label">26 关于Elasticsearch Logstash about elasticsearch logstash</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch介绍_about_elasticsearch_intro" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch介绍"/>
      <div class="microservice-label">26 关于Elasticsearch介绍 about elasticsearch intro</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch分片副本2_about_elasticsearch_shard_replica_2" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="分片副本2"/>
      <div class="microservice-label">26 关于Elasticsearch分片副本2 about elasticsearch shard replica 2</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch分片副本3_about_elasticsearch_shard_replica_3" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="分片副本3"/>
      <div class="microservice-label">26 关于Elasticsearch分片副本3 about elasticsearch shard replica 3</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch分片副本_about_elasticsearch_shard_replica" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="分片副本"/>
      <div class="microservice-label">26 关于Elasticsearch分片副本 about elasticsearch shard replica</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch场景_about_elasticsearch_scense" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch场景"/>
      <div class="microservice-label">26 关于Elasticsearch场景 about elasticsearch scense</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch排序_about_elasticsearch_sort" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch排序"/>
      <div class="microservice-label">26 关于Elasticsearch排序 about elasticsearch sort</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch排序示例_about_elasticsearch_sort_example" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch排序示例"/>
      <div class="microservice-label">26 关于Elasticsearch排序示例 about elasticsearch sort example</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch电影_about_elasticsearch_movies" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3ac.svg" alt="Elasticsearch电影"/>
      <div class="microservice-label">26 关于Elasticsearch电影 about elasticsearch movies</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch索引_about_elasticsearch_index" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch索引"/>
      <div class="microservice-label">26 关于Elasticsearch索引 about elasticsearch index</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch评估_about_elasticsearch_evaluation" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch评估"/>
      <div class="microservice-label">26 关于Elasticsearch评估 about elasticsearch evaluation</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch评估分区建议_about_elasticsearch_evaluation_partition_suggestion" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="评估分区建议"/>
      <div class="microservice-label">26 关于Elasticsearch评估分区建议 about elasticsearch evaluation partition suggestion</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/26_关于Elasticsearch集群1_about_elasticsearch_cluster_1" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch集群1"/>
      <div class="microservice-label">26 关于Elasticsearch集群1 about elasticsearch cluster 1</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/27_位图应用_bitmap_app" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="位图应用"/>
      <div class="microservice-label">27 位图应用 bitmap app</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/28_2_关于规范化形式更多2_about_normalnization_form_more_2" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="规范化2"/>
      <div class="microservice-label">28 2 关于规范化形式更多2 about normalnization form more 2</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/28_关于规范化形式更多_about_normalnization_form_more" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="规范化"/>
      <div class="microservice-label">28 关于规范化形式更多 about normalnization form more</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/29_2_关于计数器表设计_about_counter_table_design" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="计数器表"/>
      <div class="microservice-label">29 2 关于计数器表设计 about counter table design</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/29_关于缓存表摘要表_about_cahe_table_summary_tb" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="缓存表摘要"/>
      <div class="microservice-label">29 关于缓存表摘要表 about cahe table summary tb</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/30_MHA数据库和数据源切换_mha_db_and_datasource_switch" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c1.svg" alt="MHA切换"/>
      <div class="microservice-label">30 MHA数据库和数据源切换 mha db and datasource switch</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/31_Elasticsearch_Docker_elasticsearch_docker" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="Elasticsearch Docker"/>
      <div class="microservice-label">31 Elasticsearch Docker elasticsearch docker</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/32_PPC_TPC_Reactor_Proactor_ppc_tpc_reactor_proactor" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f528.svg" alt="Reactor Proactor"/>
      <div class="microservice-label">32 PPC TPC Reactor Proactor ppc tpc reactor proactor</div>
    </a>
    <a class="microservice-item" href="/#/micro_service/33_服务进程_service_process" target="_blank">
      <img class="microservice-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f528.svg" alt="服务进程"/>
      <div class="microservice-label">33 服务进程 service process</div>
    </a>
  </div>
</div>

<!-- 三高相关课题区块 -->
<style>
.high-availability-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.high-availability-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.high-availability-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.high-availability-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 220px;
  min-height: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.high-availability-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.high-availability-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
}
.high-availability-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
@media (max-width: 900px) {
  .high-availability-wall { gap: 16px; }
  .high-availability-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .high-availability-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="high-availability-section main-content">
  <div class="high-availability-title">三高相关课题</div>
  <div class="high-availability-wall">
    <a class="high-availability-item" href="/#/micro_service_pro/01_分布式ID" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="分布式ID"/>
      <div class="high-availability-label">01 分布式ID</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/02_集群配置管理-2" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c3.svg" alt="配置管理"/>
      <div class="high-availability-label">02 集群配置管理 2</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/02_集群配置管理" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c3.svg" alt="配置管理"/>
      <div class="high-availability-label">02 集群配置管理</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/03_缓存服务的访问原则" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="缓存"/>
      <div class="high-availability-label">03 缓存服务的访问原则</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/04_MQ_RPC的抉择" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e1.svg" alt="MQ/RPC"/>
      <div class="high-availability-label">04 MQ RPC的抉择</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/05_用户_个性化数据" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f464.svg" alt="用户数据"/>
      <div class="high-availability-label">05 用户 个性化数据</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/06_IP_VIP_DNS服务调用" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f310.svg" alt="DNS"/>
      <div class="high-availability-label">06 IP VIP DNS服务调用</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/07_10万并发读写的架构-车辆信息-2" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f680.svg" alt="高并发"/>
      <div class="high-availability-label">07 10万并发读写的架构 车辆信息 2</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/07_10万并发读写的架构-车辆信息-3" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f680.svg" alt="高并发"/>
      <div class="high-availability-label">07 10万并发读写的架构 车辆信息 3</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/07_10万并发读写的架构-车辆信息" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f680.svg" alt="高并发"/>
      <div class="high-availability-label">07 10万并发读写的架构 车辆信息</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/07_10万并发读写的架构" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f680.svg" alt="高并发"/>
      <div class="high-availability-label">07 10万并发读写的架构</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/08_IM_群消息投递_实时_可达" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ac.svg" alt="IM"/>
      <div class="high-availability-label">08 IM 群消息投递 实时 可达</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/09_百亿级别的Topic的架构设计" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e2.svg" alt="Topic"/>
      <div class="high-availability-label">09 百亿级别的Topic的架构设计</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/10_DeepSeek开源" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f52c.svg" alt="开源"/>
      <div class="high-availability-label">10 DeepSeek开源</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/11_ABA问题" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2753.svg" alt="ABA"/>
      <div class="high-availability-label">11 ABA问题</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/11_库存扣减策略" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4b8.svg" alt="库存"/>
      <div class="high-availability-label">11 库存扣减策略</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/12_第三方接口调用模式设计" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="接口"/>
      <div class="high-availability-label">12 第三方接口调用模式设计</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/13_架构的耦合的例子-2" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="耦合"/>
      <div class="high-availability-label">13 架构的耦合的例子 2</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/13_架构的耦合的例子" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="耦合"/>
      <div class="high-availability-label">13 架构的耦合的例子</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/14_单元化_多机房多活" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f30c.svg" alt="多活"/>
      <div class="high-availability-label">14 单元化 多机房多活</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/14_异地多活" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f30c.svg" alt="多活"/>
      <div class="high-availability-label">14 异地多活</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/15_MQ消息的幂等" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f504.svg" alt="幂等性"/>
      <div class="high-availability-label">15 MQ消息的幂等</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/16_1000万的延时任务-方案对比" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f1.svg" alt="高性能"/>
      <div class="high-availability-label">16 1000万的延时任务 方案对比</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/16_1000万的延时任务" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f1.svg" alt="高性能"/>
      <div class="high-availability-label">16 1000万的延时任务</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/17_关于搜索引擎的索引与最新数据" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="搜索引擎"/>
      <div class="high-availability-label">17 关于搜索引擎的索引与最新数据</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/18_线上变更MySqlSchema" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="Schema"/>
      <div class="high-availability-label">18 线上变更MySqlSchema</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/19_内容去重" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5d2.svg" alt="去重"/>
      <div class="high-availability-label">19 内容去重</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/20_MySql备份与恢复" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e5.svg" alt="备份恢复"/>
      <div class="high-availability-label">20 MySql备份与恢复</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/21_日志上报" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4cb.svg" alt="日志"/>
      <div class="high-availability-label">21 日志上报</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/22_DNS劫持与Https" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f512.svg" alt="DNS安全"/>
      <div class="high-availability-label">22 DNS劫持与Https</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/22_DNS劫持与ip直通车" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f512.svg" alt="DNS安全"/>
      <div class="high-availability-label">22 DNS劫持与Ip直通车</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/22_DNS的额外的用途" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f310.svg" alt="DNS"/>
      <div class="high-availability-label">22 DNS的额外的用途</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/22_企业内部自建DNSServer" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f310.svg" alt="DNS"/>
      <div class="high-availability-label">22 企业内部自建DNSServer</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/23_自制https的证书" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f511.svg" alt="证书"/>
      <div class="high-availability-label">23 自制Https的证书</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/24_分布式任务调度_真实案例" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f3.svg" alt="任务调度"/>
      <div class="high-availability-label">24 分布式任务调度 真实案例</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/24_分布式任务调度_视频转码" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f3.svg" alt="任务调度"/>
      <div class="high-availability-label">24 分布式任务调度 视频转码</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/25_高可用_故障转移_多模式" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f198.svg" alt="高可用"/>
      <div class="high-availability-label">25 高可用 故障转移 多模式</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/26_MySql的复制" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c1.svg" alt="复制"/>
      <div class="high-availability-label">26 MySql的复制</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/27_CAP就是个P" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/269b.svg" alt="CAP"/>
      <div class="high-availability-label">27 CAP就是个P</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/28_关于基础知识与规范" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="规范"/>
      <div class="high-availability-label">28 关于基础知识与规范</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/29_Kong网关项目" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e0.svg" alt="网关"/>
      <div class="high-availability-label">29 Kong网关项目</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/29_网关" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e0.svg" alt="网关"/>
      <div class="high-availability-label">29 网关</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/30_短视频_后端" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3ac.svg" alt="短视频"/>
      <div class="high-availability-label">30 短视频 后端</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/31_视频上传" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4f9.svg" alt="视频上传"/>
      <div class="high-availability-label">31 视频上传</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/32_数据服务API" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4f6.svg" alt="API"/>
      <div class="high-availability-label">32 数据服务API</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/33_大数据平台_监控" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50c.svg" alt="监控"/>
      <div class="high-availability-label">33 大数据平台 监控</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/34_容斥原理" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f52c.svg" alt="原理"/>
      <div class="high-availability-label">34 容斥原理</div>
    </a>
    <a class="high-availability-item" href="/#/micro_service_pro/36_烟草建模_mermaid" target="_blank">
      <img class="high-availability-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f33f.svg" alt="建模"/>
      <div class="high-availability-label">36 烟草建模 mermaid</div>
    </a>
  </div>
</div>

<!-- 数据库设计区块 -->
<style>
.db-design-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.db-design-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.db-design-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.db-design-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.db-design-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.db-design-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.db-design-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
}
@media (max-width: 900px) {
  .db-design-wall { gap: 16px; }
  .db-design-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .db-design-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="db-design-section main-content">
  <div class="db-design-title">数据库设计</div>
  <div class="db-design-wall">
    <a class="db-design-item" href="/#/db/README" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="文档"/>
      <div class="db-design-label">README</div>
    </a>
    <a class="db-design-item" href="/#/db/01_关于SQL注入_about_sql_injection" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e1.svg" alt="安全"/>
      <div class="db-design-label">01_关于SQL注入_about_sql_injection</div>
    </a>
    <a class="db-design-item" href="/#/db/02_关于SQL注入2_about_sql_injection_2" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6a8.svg" alt="警告"/>
      <div class="db-design-label">02_关于SQL注入2_about_sql_injection_2</div>
    </a>
    <a class="db-design-item" href="/#/db/03_关于多值_about_multi_values" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="多值"/>
      <div class="db-design-label">03_关于多值_about_multi_values</div>
    </a>
    <a class="db-design-item" href="/#/db/04_关于树_about_tree" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f333.svg" alt="树"/>
      <div class="db-design-label">04_关于树_about_tree</div>
    </a>
    <a class="db-design-item" href="/#/db/05_关于ID_about_id" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f194.svg" alt="ID"/>
      <div class="db-design-label">05_关于ID_about_id</div>
    </a>
    <a class="db-design-item" href="/#/db/06_关于引用_about_reference_" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="引用"/>
      <div class="db-design-label">06_关于引用_about_reference_</div>
    </a>
    <a class="db-design-item" href="/#/db/07_关于评估模式_about_eva_pattern" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="评估"/>
      <div class="db-design-label">07_关于评估模式_about_eva_pattern</div>
    </a>
    <a class="db-design-item" href="/#/db/08_关于多态关系_about_polymorphic_relation" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e9.svg" alt="多态"/>
      <div class="db-design-label">08_关于多态关系_about_polymorphic_relation</div>
    </a>
    <a class="db-design-item" href="/#/db/09_关于数据拆分_about_data_split" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="拆分"/>
      <div class="db-design-label">09_关于数据拆分_about_data_split</div>
    </a>
    <a class="db-design-item" href="/#/db/10_关于浮点数_about_float" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="浮点数"/>
      <div class="db-design-label">10_关于浮点数_about_float</div>
    </a>
    <a class="db-design-item" href="/#/db/11_关于枚举_about_enum" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3b2.svg" alt="枚举"/>
      <div class="db-design-label">11_关于枚举_about_enum</div>
    </a>
    <a class="db-design-item" href="/#/db/12_关于图片存储_about_image_storage" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5bc.svg" alt="图片存储"/>
      <div class="db-design-label">12_关于图片存储_about_image_storage</div>
    </a>
    <a class="db-design-item" href="/#/db/13_关于索引_about_index" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c2.svg" alt="索引"/>
      <div class="db-design-label">13_关于索引_about_index</div>
    </a>
    <a class="db-design-item" href="/#/db/13_1_关于索引基础_about_index_base" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c2.svg" alt="索引基础"/>
      <div class="db-design-label">13_1_关于索引基础_about_index_base</div>
    </a>
    <a class="db-design-item" href="/#/db/13_2_关于索引ABC_about_index_abc" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="索引ABC"/>
      <div class="db-design-label">13_2_关于索引ABC_about_index_abc</div>
    </a>
    <a class="db-design-item" href="/#/db/13_3_哈希索引_hash_index" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/0023-20e3.svg" alt="哈希索引"/>
      <div class="db-design-label">13_3_哈希索引_hash_index</div>
    </a>
    <a class="db-design-item" href="/#/db/13_4_CRC哈希自适应哈希_crc_hash_adaptive_hash" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f501.svg" alt="自适应哈希"/>
      <div class="db-design-label">13_4_CRC哈希自适应哈希_crc_hash_adaptive_hash</div>
    </a>
    <a class="db-design-item" href="/#/db/13_5_CRC哈希自适应哈希更多_crc_hash_adaptive_hash_more" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f504.svg" alt="自适应哈希更多"/>
      <div class="db-design-label">13_5_CRC哈希自适应哈希更多_crc_hash_adaptive_hash_more</div>
    </a>
    <a class="db-design-item" href="/#/db/13_6_CRC32碰撞实验_crc32_collision_lab" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a5.svg" alt="碰撞实验"/>
      <div class="db-design-label">13_6_CRC32碰撞实验_crc32_collision_lab</div>
    </a>
    <a class="db-design-item" href="/#/db/13_7_索引不仅仅是索引_index_not_just_index" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e9.svg" alt="多样性索引"/>
      <div class="db-design-label">13_7_索引不仅仅是索引_index_not_just_index</div>
    </a>
    <a class="db-design-item" href="/#/db/13_8_关于索引分析_about_index_analysis" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="索引分析"/>
      <div class="db-design-label">13_8_关于索引分析_about_index_analysis</div>
    </a>
    <a class="db-design-item" href="/#/db/13_9_关于索引更好策略_about_index_better_strages" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="更好策略"/>
      <div class="db-design-label">13_9_关于索引更好策略_about_index_better_strages</div>
    </a>
    <a class="db-design-item" href="/#/db/13_10_关于索引单列和多列差异_about_index_single_and_multi_difference" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2797.svg" alt="单列多列差异"/>
      <div class="db-design-label">13_10_关于索引单列和多列差异_about_index_single_and_multi_difference</div>
    </a>
    <a class="db-design-item" href="/#/db/13_10_索引前缀选择性_index_prefix_selectivity" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3f7.svg" alt="前缀选择性"/>
      <div class="db-design-label">13_10_索引前缀选择性_index_prefix_selectivity</div>
    </a>
    <a class="db-design-item" href="/#/db/13_11_索引MySQL解释_index_mysql_explain" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="MySQL解释"/>
      <div class="db-design-label">13_11_索引MySQL解释_index_mysql_explain</div>
    </a>
    <a class="db-design-item" href="/#/db/13_11_索引UUID_index_uuid" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f194.svg" alt="UUID"/>
      <div class="db-design-label">13_11_索引UUID_index_uuid</div>
    </a>
    <a class="db-design-item" href="/#/db/14_关于GROUP_BY_about_group_by" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f465.svg" alt="分组"/>
      <div class="db-design-label">14_关于GROUP_BY_about_group_by</div>
    </a>
    <a class="db-design-item" href="/#/db/15_关于全文搜索_about_full-text_search" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="全文搜索"/>
      <div class="db-design-label">15_关于全文搜索_about_full-text_search</div>
    </a>
    <a class="db-design-item" href="/#/db/16_关于意大利面式查询_about_yitali_spaghetti_query" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f35d.svg" alt="意大利面"/>
      <div class="db-design-label">16_关于意大利面式查询_about_yitali_spaghetti_query</div>
    </a>
    <a class="db-design-item" href="/#/db/17_关于左右内连接_about_left_right_inner_join" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f500.svg" alt="连接"/>
      <div class="db-design-label">17_关于左右内连接_about_left_right_inner_join</div>
    </a>
    <a class="db-design-item" href="/#/db/18_关于一些复杂查询_about_some_complecated_query" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ea.svg" alt="复杂查询"/>
      <div class="db-design-label">18_关于一些复杂查询_about_some_complecated_query</div>
    </a>
    <a class="db-design-item" href="/#/db/19_关于隐藏列_about_hidden_column" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f441.svg" alt="隐藏列"/>
      <div class="db-design-label">19_关于隐藏列_about_hidden_column</div>
    </a>
    <a class="db-design-item" href="/#/db/20_关于数据库结构模型_about_db_structure_model" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9f1.svg" alt="结构模型"/>
      <div class="db-design-label">20_关于数据库结构模型_about_db_structure_model</div>
    </a>
    <a class="db-design-item" href="/#/db/21_关于密码_about_password" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f511.svg" alt="密码"/>
      <div class="db-design-label">21_关于密码_about_password</div>
    </a>
    <a class="db-design-item" href="/#/db/22_关于文档_about_document" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c4.svg" alt="文档"/>
      <div class="db-design-label">22_关于文档_about_document</div>
    </a>
    <a class="db-design-item" href="/#/db/22_关于文档新_about_document_new" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f195.svg" alt="文档新"/>
      <div class="db-design-label">22_关于文档新_about_document_new</div>
    </a>
    <a class="db-design-item" href="/#/db/23_关于规范化形式_about_normalization_form" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4cf.svg" alt="规范化"/>
      <div class="db-design-label">23_关于规范化形式_about_normalization_form</div>
    </a>
    <a class="db-design-item" href="/#/db/24_关于数据库优化阶段_about_database_optimization_phases" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="优化阶段"/>
      <div class="db-design-label">24_关于数据库优化阶段_about_database_optimization_phases</div>
    </a>
    <a class="db-design-item" href="/#/db/25_关于查询优化_about_query_optimization" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="查询优化"/>
      <div class="db-design-label">25_关于查询优化_about_query_optimization</div>
    </a>
    <a class="db-design-item" href="/#/db/25_关于查询优化技术_about_query_optimization_techniques" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="查询优化技术"/>
      <div class="db-design-label">25_关于查询优化技术_about_query_optimization_techniques</div>
    </a>
    <a class="db-design-item" href="/#/db/26_关于关系代数和SQL_about_relational_algebra_and_sql" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2795.svg" alt="关系代数"/>
      <div class="db-design-label">26_关于关系代数和SQL_about_relational_algebra_and_sql</div>
    </a>
    <a class="db-design-item" href="/#/db/27_关于连接操作_about_join_operations" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="连接操作"/>
      <div class="db-design-label">27_关于连接操作_about_join_operations</div>
    </a>
    <a class="db-design-item" href="/#/db/28_关于SPJ查询优化_about_spj_query_optimization" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="SPJ查询优化"/>
      <div class="db-design-label">28_关于SPJ查询优化_about_spj_query_optimization</div>
    </a>
    <a class="db-design-item" href="/#/db/29_关于子查询优化_about_subquery_optimization" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f52c.svg" alt="子查询优化"/>
      <div class="db-design-label">29_关于子查询优化_about_subquery_optimization</div>
    </a>
    <a class="db-design-item" href="/#/db/30_关于谓词重写_about_predicate_rewrite" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/270f.svg" alt="谓词重写"/>
      <div class="db-design-label">30_关于谓词重写_about_predicate_rewrite</div>
    </a>
    <a class="db-design-item" href="/#/db/30_2_30_2" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="30_2"/>
      <div class="db-design-label">30_2_30_2</div>
    </a>
    <a class="db-design-item" href="/#/db/31_关于索引更多_about_index_more" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="索引更多"/>
      <div class="db-design-label">31_关于索引更多_about_index_more</div>
    </a>
    <a class="db-design-item" href="/#/db/32_关于多表连接_about_multi_table_join" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f517.svg" alt="多表连接"/>
      <div class="db-design-label">32_关于多表连接_about_multi_table_join</div>
    </a>
    <a class="db-design-item" href="/#/db/33_关于外连接消除_about_outer_join_elimination" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6aa.svg" alt="外连接消除"/>
      <div class="db-design-label">33_关于外连接消除_about_outer_join_elimination</div>
    </a>
    <a class="db-design-item" href="/#/db/34_关于为何需要基准测试_about_why_we_need_benchmark_test" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2753.svg" alt="基准测试"/>
      <div class="db-design-label">34_关于为何需要基准测试_about_why_we_need_benchmark_test</div>
    </a>
    <a class="db-design-item" href="/#/db/35_关于指标_about_the_metrics" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3af.svg" alt="指标"/>
      <div class="db-design-label">35_关于指标_about_the_metrics</div>
    </a>
    <a class="db-design-item" href="/#/db/36_关于基准测试工具_about_benchmark_test_tools" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9f0.svg" alt="基准测试工具"/>
      <div class="db-design-label">36_关于基准测试工具_about_benchmark_test_tools</div>
    </a>
    <a class="db-design-item" href="/#/db/36_关于设计基准测试_about_design_benchmark" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="设计基准测试"/>
      <div class="db-design-label">36_关于设计基准测试_about_design_benchmark</div>
    </a>
    <a class="db-design-item" href="/#/db/36_关于基准测试监控MySQL_about_benchmark_monitor_mysql" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ca.svg" alt="基准测试监控"/>
      <div class="db-design-label">36_关于基准测试监控MySQL_about_benchmark_monitor_mysql</div>
    </a>
    <a class="db-design-item" href="/#/db/36_关于基准测试长时间_about_benchmark_long_time" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f3.svg" alt="基准测试长时间"/>
      <div class="db-design-label">36_关于基准测试长时间_about_benchmark_long_time</div>
    </a>
    <a class="db-design-item" href="/#/db/36_关于基准测试错误视图_about_benchmark_test_error_view" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/274c.svg" alt="基准测试错误视图"/>
      <div class="db-design-label">36_关于基准测试错误视图_about_benchmark_test_error_view</div>
    </a>
    <a class="db-design-item" href="/#/db/36_关于基准测试收集系统信息_about_benchmark_collection_sysinfo" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5a5.svg" alt="收集系统信息"/>
      <div class="db-design-label">36_关于基准测试收集系统信息_about_benchmark_collection_sysinfo</div>
    </a>
    <a class="db-design-item" href="/#/db/36_1_关于基准测试建议_about_benchmark_suggestion" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="基准测试建议"/>
      <div class="db-design-label">36_1_关于基准测试建议_about_benchmark_suggestion</div>
    </a>
    <a class="db-design-item" href="/#/db/37_0_关于性能_about_performance" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26a1.svg" alt="性能"/>
      <div class="db-design-label">37_0_关于性能_about_performance</div>
    </a>
    <a class="db-design-item" href="/#/db/37_0_关于性能2_about_performance_2" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26a1.svg" alt="性能2"/>
      <div class="db-design-label">37_0_关于性能2_about_performance_2</div>
    </a>
    <a class="db-design-item" href="/#/db/37_1_关于性能MySQL_show_status_about_perform_mysql_show_status" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ca.svg" alt="MySQL show status"/>
      <div class="db-design-label">37_1_关于性能MySQL_show_status_about_perform_mysql_show_status</div>
    </a>
    <a class="db-design-item" href="/#/db/37_2_关于使用Profile分析性能_about_perform_using_profile" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="Profile分析性能"/>
      <div class="db-design-label">37_2_关于使用Profile分析性能_about_perform_using_profile</div>
    </a>
    <a class="db-design-item" href="/#/db/37_3_关于性能Show_Profile_about_performance_show_profile" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="Show Profile"/>
      <div class="db-design-label">37_3_关于性能Show_Profile_about_performance_show_profile</div>
    </a>
    <a class="db-design-item" href="/#/db/37_4_关于性能慢查询日志_about_performance_slow_query_log" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f422.svg" alt="慢查询日志"/>
      <div class="db-design-label">37_4_关于性能慢查询日志_about_performance_slow_query_log</div>
    </a>
    <a class="db-design-item" href="/#/db/37_5_关于进程列表监控_about_processlist_monitor" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4cb.svg" alt="进程列表监控"/>
      <div class="db-design-label">37_5_关于进程列表监控_about_processlist_monitor</div>
    </a>
    <a class="db-design-item" href="/#/db/37_6_关于sysbench监控_about_sysbench_monitor" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="sysbench监控"/>
      <div class="db-design-label">37_6_关于sysbench监控_about_sysbench_monitor</div>
    </a>
    <a class="db-design-item" href="/#/db/37_7_关于可视化进程列表_about_visual_processlist" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f441.svg" alt="可视化进程列表"/>
      <div class="db-design-label">37_7_关于可视化进程列表_about_visual_processlist</div>
    </a>
    <a class="db-design-item" href="/#/db/37_8_关于MySQL中的慢查询_about_slow_query_in_mysql" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f422.svg" alt="慢查询"/>
      <div class="db-design-label">37_8_关于MySQL中的慢查询_about_slow_query_in_mysql</div>
    </a>
    <a class="db-design-item" href="/#/db/37_9_关于使用pt-query-digest分析慢查询_about_slow_query_using_pt-query-digest" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f52c.svg" alt="pt-query-digest分析慢查询"/>
      <div class="db-design-label">37_9_关于使用pt-query-digest分析慢查询_about_slow_query_using_pt-query-digest</div>
    </a>
    <a class="db-design-item" href="/#/db/37_10_关于MySQL慢查询_about_mysql_slow_query" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f422.svg" alt="MySQL慢查询"/>
      <div class="db-design-label">37_10_关于MySQL慢查询_about_mysql_slow_query</div>
    </a>
    <a class="db-design-item" href="/#/db/37_11_压力测试与sysbench_pression_test_with_sysbench" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3cb.svg" alt="压力测试"/>
      <div class="db-design-label">37_11_压力测试与sysbench_pression_test_with_sysbench</div>
    </a>
    <a class="db-design-item" href="/#/db/38_关于数据类型_about_data_types" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="数据类型"/>
      <div class="db-design-label">38_关于数据类型_about_data_types</div>
    </a>
    <a class="db-design-item" href="/#/db/38_2_关于电商中的数据类型_about_data_type_in_ecomic" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6d2.svg" alt="电商数据类型"/>
      <div class="db-design-label">38_2_关于电商中的数据类型_about_data_type_in_ecomic</div>
    </a>
    <a class="db-design-item" href="/#/db/38_3_关于文本_blob和临时文件排序文件_about_text_blob_and_tempfile_sortfile" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c4.svg" alt="文本和临时文件"/>
      <div class="db-design-label">38_3_关于文本_blob和临时文件排序文件_about_text_blob_and_tempfile_sortfile</div>
    </a>
    <a class="db-design-item" href="/#/db/38_4_关于日期类型更多_about_date_type_more" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c5.svg" alt="日期类型更多"/>
      <div class="db-design-label">38_4_关于日期类型更多_about_date_type_more</div>
    </a>
    <a class="db-design-item" href="/#/db/38_5_时区和日期时间_zone_and_datetime" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f310.svg" alt="时区和日期时间"/>
      <div class="db-design-label">38_5_时区和日期时间_zone_and_datetime</div>
    </a>
    <a class="db-design-item" href="/#/db/38_6_时区和日期时间更多_zone_and_datetime_more" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f30d.svg" alt="时区和日期时间更多"/>
      <div class="db-design-label">38_6_时区和日期时间更多_zone_and_datetime_more</div>
    </a>
    <a class="db-design-item" href="/#/db/39_关于修改表_about_modify_table" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/270f.svg" alt="修改表"/>
      <div class="db-design-label">39_关于修改表_about_modify_table</div>
    </a>
    <a class="db-design-item" href="/#/db/40_1_关于日常维护_about_daily_maintenance" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9f9.svg" alt="日常维护"/>
      <div class="db-design-label">40_1_关于日常维护_about_daily_maintenance</div>
    </a>
    <a class="db-design-item" href="/#/db/40_2_关于日常维护_about_daily_maintenance" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9f9.svg" alt="日常维护"/>
      <div class="db-design-label">40_2_关于日常维护_about_daily_maintenance</div>
    </a>
    <a class="db-design-item" href="/#/db/40_3_关于日常维护_about_daily_maintenance" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9f9.svg" alt="日常维护"/>
      <div class="db-design-label">40_3_关于日常维护_about_daily_maintenance</div>
    </a>
    <a class="db-design-item" href="/#/db/41_1_数据迁移_data_migration" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f69a.svg" alt="数据迁移"/>
      <div class="db-design-label">41_1_数据迁移_data_migration</div>
    </a>
    <a class="db-design-item" href="/#/db/42_关于B树_about_btree" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f333.svg" alt="B树"/>
      <div class="db-design-label">42_关于B树_about_btree</div>
    </a>
    <a class="db-design-item" href="/#/db/43_关于GIS示例_about_gis_example" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5fa.svg" alt="GIS示例"/>
      <div class="db-design-label">43_关于GIS示例_about_gis_example</div>
    </a>
    <a class="db-design-item" href="/#/db/44_关于分片分析_about_sharding_analysis" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e9.svg" alt="分片分析"/>
      <div class="db-design-label">44_关于分片分析_about_sharding_analysis</div>
    </a>
    <a class="db-design-item" href="/#/db/45_时序数据库介绍_timeserial_db_intro" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f1.svg" alt="时序数据库"/>
      <div class="db-design-label">45_时序数据库介绍_timeserial_db_intro</div>
    </a>
    <a class="db-design-item" href="/#/db/46_1_查询拆分_query_split" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2702.svg" alt="查询拆分"/>
      <div class="db-design-label">46_1_查询拆分_query_split</div>
    </a>
    <a class="db-design-item" href="/#/db/46_2_查询解释成本_query_explain_cost" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4b0.svg" alt="查询解释成本"/>
      <div class="db-design-label">46_2_查询解释成本_query_explain_cost</div>
    </a>
    <a class="db-design-item" href="/#/db/46_3_查询优化器为何错误_query_optimizer_why_wrong" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2753.svg" alt="为何错误"/>
      <div class="db-design-label">46_3_查询优化器为何错误_query_optimizer_why_wrong</div>
    </a>
    <a class="db-design-item" href="/#/db/46_4_查询优化器静态和动态_query_optimizer_static_and_dymanic" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f501.svg" alt="静态和动态"/>
      <div class="db-design-label">46_4_查询优化器静态和动态_query_optimizer_static_and_dymanic</div>
    </a>
    <a class="db-design-item" href="/#/db/46_5_查询优化器限制_query_optimizer_limit" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26d4.svg" alt="限制"/>
      <div class="db-design-label">46_5_查询优化器限制_query_optimizer_limit</div>
    </a>
    <a class="db-design-item" href="/#/db/46_5_1_查询优化器限制2_query_optimizer_limit_2" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/26d4.svg" alt="限制2"/>
      <div class="db-design-label">46_5_1_查询优化器限制2_query_optimizer_limit_2</div>
    </a>
    <a class="db-design-item" href="/#/db/46_6_查询优化器示例_query_optimizer_samples" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4cb.svg" alt="示例"/>
      <div class="db-design-label">46_6_查询优化器示例_query_optimizer_samples</div>
    </a>
    <a class="db-design-item" href="/#/db/46_7_计数_count" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="计数"/>
      <div class="db-design-label">46_7_计数_count</div>
    </a>
    <a class="db-design-item" href="/#/db/46_8_计数差异_count_difference" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2796.svg" alt="计数差异"/>
      <div class="db-design-label">46_8_计数差异_count_difference</div>
    </a>
    <a class="db-design-item" href="/#/db/47_1_位置_locations" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4cd.svg" alt="位置"/>
      <div class="db-design-label">47_1_位置_locations</div>
    </a>
    <a class="db-design-item" href="/#/db/48_1_关于配置内存_about_config_memory" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="配置内存"/>
      <div class="db-design-label">48_1_关于配置内存_about_config_memory</div>
    </a>
    <a class="db-design-item" href="/#/db/48_2_CPU选择_cpu_choice" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5a5.svg" alt="CPU选择"/>
      <div class="db-design-label">48_2_CPU选择_cpu_choice</div>
    </a>
    <a class="db-design-item" href="/#/db/49_1_复制主从介绍_replication_master_slave_intro" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f501.svg" alt="复制主从"/>
      <div class="db-design-label">49_1_复制主从介绍_replication_master_slave_intro</div>
    </a>
    <a class="db-design-item" href="/#/db/49_2_复制介绍_replication_intro" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f501.svg" alt="复制介绍"/>
      <div class="db-design-label">49_2_复制介绍_replication_intro</div>
    </a>
    <a class="db-design-item" href="/#/db/49_3_复制行和SQL_replication_row_and_sql" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c4.svg" alt="复制行和SQL"/>
      <div class="db-design-label">49_3_复制行和SQL_replication_row_and_sql</div>
    </a>
    <a class="db-design-item" href="/#/db/50_db_crash_scense" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a5.svg" alt="crash"/>
      <div class="db-design-label">50_db_crash_scense</div>
    </a>
    <a class="db-design-item" href="/#/db/50_2_崩溃分析_crash_analysis" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e1.svg" alt="崩溃分析"/>
      <div class="db-design-label">50_2_崩溃分析_crash_analysis</div>
    </a>
    <a class="db-design-item" href="/#/db/51_1_高可用MTBF_ha_mtbf" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f1.svg" alt="MTBF"/>
      <div class="db-design-label">51_1_高可用MTBF_ha_mtbf</div>
    </a>
    <a class="db-design-item" href="/#/db/51_2_高可用MTTR_ha_mttr" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f3.svg" alt="MTTR"/>
      <div class="db-design-label">51_2_高可用MTTR_ha_mttr</div>
    </a>
    <a class="db-design-item" href="/#/db/52_1_磁盘扩展Ceph_disk_expand_ceph" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bd.svg" alt="磁盘扩展"/>
      <div class="db-design-label">52_1_磁盘扩展Ceph_disk_expand_ceph</div>
    </a>
    <a class="db-design-item" href="/#/db/53_ECrop电话号码_ecrop_phonenumber" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4de.svg" alt="电话号码"/>
      <div class="db-design-label">53_ECrop电话号码_ecrop_phonenumber</div>
    </a>
    <a class="db-design-item" href="/#/db/54_1_备份介绍_backup_intro" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="备份介绍"/>
      <div class="db-design-label">54_1_备份介绍_backup_intro</div>
    </a>
    <a class="db-design-item" href="/#/db/54_2_备份定义恢复_backup_define_recover" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/267b.svg" alt="定义恢复"/>
      <div class="db-design-label">54_2_备份定义恢复_backup_define_recover</div>
    </a>
    <a class="db-design-item" href="/#/db/54_3_备份在线离线_backup_online_offline" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f310.svg" alt="在线离线"/>
      <div class="db-design-label">54_3_备份在线离线_backup_online_offline</div>
    </a>
    <a class="db-design-item" href="/#/db/54_4_备份逻辑备份_backup_logic_backup" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e0.svg" alt="逻辑备份"/>
      <div class="db-design-label">54_4_备份逻辑备份_backup_logic_backup</div>
    </a>
    <a class="db-design-item" href="/#/db/54_5_备份物理备份_backup_pysical_backup" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c4.svg" alt="物理备份"/>
      <div class="db-design-label">54_5_备份物理备份_backup_pysical_backup</div>
    </a>
    <a class="db-design-item" href="/#/db/54_5_备份应备份什么_backup_what_should_be_bk" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2753.svg" alt="应备份什么"/>
      <div class="db-design-label">54_5_备份应备份什么_backup_what_should_be_bk</div>
    </a>
    <a class="db-design-item" href="/#/db/54_6_增量和差异备份_incremental_and_different_bk" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="增量和差异备份"/>
      <div class="db-design-label">54_6_增量和差异备份_incremental_and_different_bk</div>
    </a>
    <a class="db-design-item" href="/#/db/54_7_备份二进制备份_backup_binary_bk" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4be.svg" alt="二进制备份"/>
      <div class="db-design-label">54_7_备份二进制备份_backup_binary_bk</div>
    </a>
    <a class="db-design-item" href="/#/db/54_8_备份和恢复示例_backup_and_recovery_sample" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f501.svg" alt="备份和恢复示例"/>
      <div class="db-design-label">54_8_备份和恢复示例_backup_and_recovery_sample</div>
    </a>
    <a class="db-design-item" href="/#/db/54_9_备份和恢复100G示例_backup_and_recovery_100g_sample" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="100G示例"/>
      <div class="db-design-label">54_9_备份和恢复100G示例_backup_and_recovery_100g_sample</div>
    </a>
    <a class="db-design-item" href="/#/db/55_索引生成器介绍_index_gener_intro" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2699.svg" alt="索引生成器"/>
      <div class="db-design-label">55_索引生成器介绍_index_gener_intro</div>
    </a>
    <a class="db-design-item" href="/#/db/56_索引普通类型介绍_index_normal_types_intro" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="索引普通类型"/>
      <div class="db-design-label">56_索引普通类型介绍_index_normal_types_intro</div>
    </a>
    <a class="db-design-item" href="/#/db/56_2_索引普通类型介绍_index_normal_types_intro" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="索引普通类型2"/>
      <div class="db-design-label">56_2_索引普通类型介绍_index_normal_types_intro</div>
    </a>
    <a class="db-design-item" href="/#/db/57_1_设计年龄_design_ages" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f382.svg" alt="设计年龄"/>
      <div class="db-design-label">57_1_设计年龄_design_ages</div>
    </a>
    <a class="db-design-item" href="/#/db/57_2_设计示例_design_example" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4dd.svg" alt="设计示例"/>
      <div class="db-design-label">57_2_设计示例_design_example</div>
    </a>
    <a class="db-design-item" href="/#/db/57_3_设计JSON_design_json" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="设计JSON"/>
      <div class="db-design-label">57_3_设计JSON_design_json</div>
    </a>
    <a class="db-design-item" href="/#/db/58_向量数据库_vector_db" target="_blank">
      <img class="db-design-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9ec.svg" alt="向量数据库"/>
      <div class="db-design-label">58_向量数据库_vector_db</div>
    </a>
  </div>
</div>

<!-- 系统建设基础知识区块 -->
<style>
.sys-knowledge-section {
  margin: 64px 0 48px 0;
  background: #fff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #222;
  text-align: center;
}
.sys-knowledge-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.sys-knowledge-carousel {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  max-width: 1000px;
  margin: 0 auto;
  position: relative;
}
.sys-arrow {
  width: 48px;
  height: 48px;
  background: rgba(0,255,231,0.12);
  border: none;
  border-radius: 50%;
  color: #00ffe7;
  font-size: 2em;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
  z-index: 2;
}
.sys-arrow:hover {
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  color: #003a5d;
}
.sys-carousel-img {
  max-width: 1600px;
  max-height: 900px;
  width: 100%;
  border-radius: 16px;
  box-shadow: 0 4px 24px rgba(0,255,231,0.10);
  background: #f2f6fa;
  object-fit: contain;
}
@media (max-width: 1800px) {
  .sys-carousel-img { max-width: 98vw; max-height: 60vw; }
  .sys-knowledge-carousel { gap: 8px; }
}
@media (max-width: 1300px) {
  .sys-carousel-img { max-width: 98vw; max-height: 48vw; }
}
@media (max-width: 900px) {
  .sys-carousel-img { max-width: 98vw; max-height: 40vw; }
}
@media (max-width: 600px) {
  .sys-carousel-img { max-width: 98vw; max-height: 220px; }
  .sys-arrow { width: 32px; height: 32px; font-size: 1.2em; }
}
</style>
<div class="sys-knowledge-section main-content">
  <div class="sys-knowledge-title">分布式服务架构基础知识</div>
  <div class="sys-knowledge-carousel">
    <button class="sys-arrow" id="sys-arrow-left">&#8592;</button>
    <img id="sys-carousel-img" class="sys-carousel-img" src="images/microservicea/1-4种RestAPI的认证方法.jpeg" alt="系统建设基础知识"/>
    <button class="sys-arrow" id="sys-arrow-right">&#8594;</button>
  </div>
</div>
<!-- 轮播JS请放在index.html，图片列表可在JS中维护或自动生成 -->

<!-- 数字化转型的区块 -->
<div class="main-content" style="margin: 64px auto 0 auto;">
    <h1 style="font-size: 2.8em; font-weight: bold; margin-bottom: 0.3em; background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI & 数字化转型</h1>
        <div style="font-size: 1.3em; margin-bottom: 1.5em; color: #b6eaff;">充分利用数据和人工智能，加速企业变革</div>
  <div style="background: #fff; border-radius: 18px; box-shadow: 0 4px 32px rgba(0,0,0,0.06); padding: 48px 0 32px 0;">
    <div class="feature-grid" style="background: none; box-shadow: none; border-radius: 0; padding: 0; margin: 0;">
      <div class="feature-item">
        <img src="images/corps/project-manage.png" alt="项目管理">
        <p>项目管理</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/customer-manage.png" alt="客户管理">
        <p>客户管理</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/production-manage.png" alt="生产管理">
        <p>生产管理</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/quality-control.png" alt="质量控制">
        <p>质量控制</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/plm-product-lifecycle.png" alt="产品生命周期管理">
        <p>产品生命周期</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/invoicing-2.0.png" alt="发票管理">
        <p>发票管理</p>
      </div>
       <div class="feature-item">
        <img src="images/corps/asset-management.png" alt="资产管理">
        <p>资产管理</p>
      </div>
       <div class="feature-item">
        <img src="images/corps/process-manage.png" alt="流程管理">
        <p>流程管理</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/after-manage.png" alt="售后管理">
        <p>售后管理</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/functionality-updates.png" alt="资产管理">
        <p>资产管理</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/internet-management.png" alt="研发管理">
        <p>研发管理</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/oa-coworking-platform.png" alt="OA协同平台">
        <p>OA协同平台</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/okr-manage-2.0.png" alt="OKR管理">
        <p>OKR管理</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/section2-1-1.png" alt="设备巡检">
        <p>设备巡检</p>
      </div>
      <div class="feature-item">
        <img src="images/corps/section2-1-4.png" alt="人事管理">
        <p>人事管理</p>
      </div>
    </div>
    <p align="center" style="margin-top: 50px;">
      <a href="#entropic-consulting" style="text-decoration: none;">
        <button style="background-color: #4CAF50; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border: none; border-radius: 8px;">
          了解更多
        </button>
      </a>
    </p>
  </div>
</div>

<!-- 大数据平台区块 -->
<style>
.bigdata-section {
  margin: 64px 0 48px 0;
  background: #fff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 32px 0 32px 0;
  color: #222;
}
.bigdata-title {
  font-size: 2em;
  font-weight: bold;
  text-align: center;
  margin-bottom: 24px;
  color: #003a5d;
}
.bigdata-main {
  display: flex;
  flex-wrap: wrap;
  max-width: 1200px;
  margin: 0 auto;
  min-height: 340px;
}
.bigdata-tabs {
  flex: 0 0 180px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  justify-content: flex-start;
  align-items: flex-start;
  z-index: 2;
  margin-top: 12px;
}
.bigdata-tab {
  width: 140px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  border-radius: 12px;
  background: #f2f6fa;
  color: #003a5d;
  font-size: 1.08em;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.3s, color 0.3s;
  margin-bottom: 8px;
  position: relative;
  border: 1px solid #e0eaff;
  padding-left: 18px;
}
.bigdata-tab.active {
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  color: #003a5d;
  font-weight: bold;
  border: 1px solid #00ffe7;
}
.bigdata-content {
  flex: 1;
  min-width: 0;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: flex-start;
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.03);
  margin-left: 24px;
  padding: 0 24px;
}
.bigdata-detail {
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: flex-start;
  background: #fff;
}
.bigdata-detail .bd-title {
  font-size: 1.5em;
  font-weight: bold;
  margin-bottom: 12px;
  color: #003a5d;
  margin-top: 12px;
}
.bigdata-detail .bd-desc {
  font-size: 1.08em;
  color: #2a3a4d;
  margin-bottom: 32px;
  line-height: 1.8;
  margin-top: 0;
}
.bigdata-detail .bd-img {
  width: 100%;
  max-width: 95%;
  border-radius: 16px;
  box-shadow: 0 4px 32px rgba(0,255,231,0.10);
  background: #f2f6fa;
  margin: 0 auto 0 auto;
  display: block;
}
@media (max-width: 900px) {
  .bigdata-main { flex-direction: column; }
  .bigdata-tabs { flex-direction: row; flex: none; margin-bottom: 18px; margin-top: 0; align-items: stretch; }
  .bigdata-tab { margin-bottom: 0; margin-right: 8px; width: auto; min-width: 100px; justify-content: center; padding-left: 0; }
  .bigdata-content { min-height: 320px; margin-left: 0; padding: 0 8px; }
  .bigdata-detail { padding: 0; max-width: 100%; }
  .bigdata-detail .bd-img { max-width: 100%; }
}
</style>
<div class="bigdata-section main-content">
  <div class="bigdata-title">大数据平台</div>
  <div class="bigdata-main">
    <div class="bigdata-tabs">
      <div class="bigdata-tab active" id="bigdata-tab-0" onclick="showBigdata(0)">大数据治理概念</div>
      <div class="bigdata-tab" id="bigdata-tab-1" onclick="showBigdata(1)">大数据治理方法论</div>
      <div class="bigdata-tab" id="bigdata-tab-2" onclick="showBigdata(2)">大数据治理服务体系</div>
      <div class="bigdata-tab" id="bigdata-tab-3" onclick="showBigdata(3)">大数据治理解决方案</div>
      <div class="bigdata-tab" id="bigdata-tab-4" onclick="showBigdata(4)">数据应用成熟度评估</div>
      <div class="bigdata-tab" id="bigdata-tab-5" onclick="showBigdata(5)">数据仓库</div>
      <div class="bigdata-tab" id="bigdata-tab-6" onclick="showBigdata(6)">数据分层</div>
      <div class="bigdata-tab" id="bigdata-tab-7" onclick="showBigdata(7)">维度-度量-指标</div>
      <div class="bigdata-tab" id="bigdata-tab-8" onclick="showBigdata(8)">Lambda架构</div>
      <div class="bigdata-tab" id="bigdata-tab-9" onclick="showBigdata(9)">Kappa架构</div>
      <div class="bigdata-tab" id="bigdata-tab-10" onclick="showBigdata(10)">混合架构</div>
      <div class="bigdata-tab" id="bigdata-tab-11" onclick="showBigdata(11)">产品架构</div>
      <div class="bigdata-tab" id="bigdata-tab-12" onclick="showBigdata(12)">技术架构</div>
      <div class="bigdata-tab" id="bigdata-tab-13" onclick="showBigdata(13)">电商大屏</div>
      <div class="bigdata-tab" id="bigdata-tab-14" onclick="showBigdata(14)">销售大屏</div>
      <div class="bigdata-tab" id="bigdata-tab-15" onclick="showBigdata(15)">轨迹</div>
      <div class="bigdata-tab" id="bigdata-tab-16" onclick="showBigdata(16)">Superset</div>
      <div class="bigdata-tab" id="bigdata-tab-17" onclick="showBigdata(17)">Redash</div>
      <div class="bigdata-tab" id="bigdata-tab-18" onclick="showBigdata(18)">Metabase</div>
      <div class="bigdata-tab" id="bigdata-tab-19" onclick="showBigdata(19)">Kettle</div>
      <div class="bigdata-tab" id="bigdata-tab-20" onclick="showBigdata(20)">Presto</div>
      <div class="bigdata-tab" id="bigdata-tab-21" onclick="showBigdata(21)">Impala</div>
      <div class="bigdata-tab" id="bigdata-tab-22" onclick="showBigdata(22)">Kylin</div>
      <div class="bigdata-tab" id="bigdata-tab-23" onclick="showBigdata(23)">Airflow</div>
      <div class="bigdata-tab" id="bigdata-tab-24" onclick="showBigdata(24)">Oozie</div>
      <div class="bigdata-tab" id="bigdata-tab-25" onclick="showBigdata(25)">Sqoop&Flume</div>
      <div class="bigdata-tab" id="bigdata-tab-26" onclick="showBigdata(26)">数据血缘</div>
    </div>
    <div class="bigdata-content">
      <div class="bigdata-detail" id="bigdata-detail">
        <div class="bd-title">数据仓库</div>
        <div class="bd-desc">数据仓库是一个面向主题的（Subject Oriented）、集成的（Integrate）、相对稳定的（Non-Volatile）、反映历史变化（Time Variant）的数据集合，用于支持管理决策</div>
        <img class="bd-img" src="images/bigdata/bigdata_architecture_1.png" alt="数据仓库"/>
      </div>
    </div>
  </div>
</div>

<!-- 大数据平台之大屏区块 -->
<style>
.dashboard-gallery-section {
  margin: 64px 0 48px 0;
  background: #0f1c3a;
  border-radius: 18px;
  padding: 40px 0 40px 0;
  color: #fff;
  text-align: center;
  box-shadow: 0 4px 32px rgba(0,0,0,0.10);
}
.dashboard-gallery-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.dashboard-gallery {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 32px;
  max-width: 1200px;
  margin: 0 auto;
}
.dashboard-main-img {
  width: 100%;
  max-width: 900px;
  height: 480px;
  object-fit: contain;
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(0,255,231,0.10);
  background: #fff;
  transition: box-shadow 0.3s, transform 0.3s;
}
.dashboard-thumbs {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  justify-content: center;
  margin-top: 12px;
  max-width: 900px;
}
.dashboard-thumb {
  width: 100px;
  height: 60px;
  object-fit: cover;
  border-radius: 8px;
  cursor: pointer;
  border: 2px solid transparent;
  transition: border 0.2s, transform 0.2s;
  background: #fff;
  opacity: 0.7;
}
.dashboard-thumb.active {
  border: 2px solid #00ffe7;
  opacity: 1;
  transform: scale(1.08);
  z-index: 2;
}
@media (max-width: 1000px) {
  .dashboard-main-img { max-width: 98vw; height: 36vw; min-height: 180px; }
  .dashboard-thumbs { max-width: 98vw; }
}
@media (max-width: 600px) {
  .dashboard-main-img { height: 120px; }
  .dashboard-thumb { width: 60px; height: 36px; }
}
.dashboard-init-mask {
  width: 100%;
  height: 480px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  background: #12263a;
  border-radius: 18px;
  cursor: pointer;
  z-index: 2;
}
</style>
<div class="dashboard-gallery-section main-content">
  <div class="dashboard-gallery-title">大数据平台-驾驶舱（大屏）（118个案例）</div>
  <div class="dashboard-gallery">
    <div id="dashboard-init-mask" class="dashboard-init-mask">
      <span style="color:#00ffe7;font-size:1.5em;">点击加载大屏轮播</span>
    </div>
    <img id="dashboard-main-img" class="dashboard-main-img" src="" alt="大屏预览" style="display:none;"/>
    <div class="dashboard-thumbs" id="dashboard-thumbs" style="display:none;"></div>
  </div>
</div>

<!-- 大数据治理（平台）建设知识 -->
<style>
.bigdata-section {
  margin: 64px 0 48px 0;
  background: #f8fbff;
  border-radius: 18px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.06);
  padding: 40px 0 40px 0;
  color: #003a5d;
  text-align: center;
}
.bigdata-title {
  font-size: 2.2em;
  font-weight: bold;
  margin-bottom: 24px;
  background: linear-gradient(90deg, #00ffe7 0%, #1ec8ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.bigdata-wall {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1400px;
  margin: 0 auto;
}
.bigdata-item {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0,255,231,0.08);
  width: 260px;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 28px 18px 18px 18px;
  transition: box-shadow 0.2s, transform 0.2s;
  cursor: pointer;
  text-decoration: none;
  color: #003a5d;
  position: relative;
}
.bigdata-item:hover {
  box-shadow: 0 8px 32px rgba(0,255,231,0.18);
  transform: translateY(-6px) scale(1.04);
  color: #1ec8ff;
}
.bigdata-label {
  font-size: 1.05em;
  font-weight: bold;
  margin-bottom: 8px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.bigdata-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  display: block;
}
@media (max-width: 900px) {
  .bigdata-wall { gap: 16px; }
  .bigdata-item { width: 44vw; min-width: 140px; }
}
@media (max-width: 600px) {
  .bigdata-item { width: 98vw; min-width: 100px; }
}
</style>
<div class="bigdata-section main-content">
  <div class="bigdata-title">大数据平台建设</div>
  <div class="bigdata-wall">
    <a class="bigdata-item" href="/#/bigdata/01_日志采集" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e6.svg" alt="日志采集"/>
      <div class="bigdata-label">01_日志采集</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/02_数据漂移" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f504.svg" alt="数据漂移"/>
      <div class="bigdata-label">02_数据漂移</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/03_统一计算平台建设" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f6e0.svg" alt="计算平台"/>
      <div class="bigdata-label">03_统一计算平台建设</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/03_大数据平台建设工具集" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="工具集"/>
      <div class="bigdata-label">03_大数据平台建设工具集</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/03_大数据平台构建建议" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="构建建议"/>
      <div class="bigdata-label">03_大数据平台构建建议</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/04_大数据平台建设_数据服务" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2601.svg" alt="数据服务"/>
      <div class="bigdata-label">04_大数据平台建设_数据服务</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/04_CDH" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c3.svg" alt="CDH"/>
      <div class="bigdata-label">04_CDH</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/05_电影数据面板" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3ac.svg" alt="电影数据"/>
      <div class="bigdata-label">05_电影数据面板</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/06_电影相关的LineChart" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="LineChart"/>
      <div class="bigdata-label">06_电影相关的LineChart</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/07_博物馆_Analysis" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3db.svg" alt="博物馆分析"/>
      <div class="bigdata-label">07_博物馆_Analysis</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/07_博物馆DDL" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c4.svg" alt="博物馆DDL"/>
      <div class="bigdata-label">07_博物馆DDL</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/07_博物馆DDL_2" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c4.svg" alt="博物馆DDL2"/>
      <div class="bigdata-label">07_博物馆DDL_2</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/08_构建简单的图视图" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5fa.svg" alt="图视图"/>
      <div class="bigdata-label">08_构建简单的图视图</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/09_superset_connect_to_mysql" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="Superset连接MySQL"/>
      <div class="bigdata-label">09_superset_connect_to_mysql</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/10_flask_cors" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f40d.svg" alt="flask_cors"/>
      <div class="bigdata-label">10_flask_cors</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/10_数据生命周期_data_lifesycle" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5d3.svg" alt="数据生命周期"/>
      <div class="bigdata-label">10_数据生命周期_data_lifesycle</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/10_2_数据生命周期工具_data_lifesycle_tools" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="生命周期工具"/>
      <div class="bigdata-label">10_2_数据生命周期工具_data_lifesycle_tools</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/11_使用DBScan识别异常流量" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="DBScan异常流量"/>
      <div class="bigdata-label">11_使用DBScan识别异常流量</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/12_MapBox" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5fa.svg" alt="MapBox"/>
      <div class="bigdata-label">12_MapBox</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/12_MapBox的跨域请求问题" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f310.svg" alt="MapBox跨域"/>
      <div class="bigdata-label">12_MapBox的跨域请求问题</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/14_使用Kettle构建ETL" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f528.svg" alt="Kettle ETL"/>
      <div class="bigdata-label">14_使用Kettle构建ETL</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/15_初步了解nifi" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="初步了解nifi"/>
      <div class="bigdata-label">15_初步了解nifi</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/15_进一步了解nifi" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4a1.svg" alt="进一步了解nifi"/>
      <div class="bigdata-label">15_进一步了解nifi</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/15_使用nifi进行数据抽取场景" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50c.svg" alt="nifi数据抽取"/>
      <div class="bigdata-label">15_使用nifi进行数据抽取场景</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/16_airflow" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f32c.svg" alt="airflow"/>
      <div class="bigdata-label">16_airflow</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/16_airflow_README" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="airflow_README"/>
      <div class="bigdata-label">16_airflow_README</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/16_airflow_setup_README" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f527.svg" alt="airflow_setup_README"/>
      <div class="bigdata-label">16_airflow_setup_README</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/18_hasura_api_构建_README" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4bb.svg" alt="hasura_api_构建"/>
      <div class="bigdata-label">18_hasura_api_构建_README</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/19_calcite_sql_网关_多数据源引擎协调" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f309.svg" alt="calcite_sql_网关"/>
      <div class="bigdata-label">19_calcite_sql_网关_多数据源引擎协调</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/20_数据服务化API" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4f6.svg" alt="数据服务化API"/>
      <div class="bigdata-label">20_数据服务化API</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/20_数据服务化API_2" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4f6.svg" alt="数据服务化API2"/>
      <div class="bigdata-label">20_数据服务化API_2</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/20_数据服务化API_wso2" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4f6.svg" alt="数据服务化APIwso2"/>
      <div class="bigdata-label">20_数据服务化API_wso2</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/21_oozie_编排与调度" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/23f3.svg" alt="oozie编排调度"/>
      <div class="bigdata-label">21_oozie_编排与调度</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/22_sqoop_flume" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e7.svg" alt="sqoop_flume"/>
      <div class="bigdata-label">22_sqoop_flume</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/23_数据分层" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c2.svg" alt="数据分层"/>
      <div class="bigdata-label">23_数据分层</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/24_指标-维度-度量" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="指标维度度量"/>
      <div class="bigdata-label">24_指标-维度-度量</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/24_指标-维度-度量-v2" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f522.svg" alt="指标维度度量v2"/>
      <div class="bigdata-label">24_指标-维度-度量-v2</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/movie_ddl" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3a5.svg" alt="movie_ddl"/>
      <div class="bigdata-label">movie_ddl</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/README" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="README"/>
      <div class="bigdata-label">README</div>
    </a>
    <!-- bigdata/components 子目录md文件 -->
    <a class="bigdata-item" href="/#/bigdata/components/superset" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="superset"/>
      <div class="bigdata-label">components/superset</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/airflow" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f32c.svg" alt="airflow"/>
      <div class="bigdata-label">components/airflow</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/openmetadata" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c1.svg" alt="openmetadata"/>
      <div class="bigdata-label">components/openmetadata</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/great_expectations" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f60e.svg" alt="great_expectations"/>
      <div class="bigdata-label">components/great_expectations</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/atlas" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5fa.svg" alt="atlas"/>
      <div class="bigdata-label">components/atlas</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/ranger" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f98c.svg" alt="ranger"/>
      <div class="bigdata-label">components/ranger</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/kafka_connect" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e1.svg" alt="kafka_connect"/>
      <div class="bigdata-label">components/kafka_connect</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/rest_api" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4f6.svg" alt="rest_api"/>
      <div class="bigdata-label">components/rest_api</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/zeppelin" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2708.svg" alt="zeppelin"/>
      <div class="bigdata-label">components/zeppelin</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/presto_trino" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3c1.svg" alt="presto_trino"/>
      <div class="bigdata-label">components/presto_trino</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/hive" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f41d.svg" alt="hive"/>
      <div class="bigdata-label">components/hive</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/flink" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f41f.svg" alt="flink"/>
      <div class="bigdata-label">components/flink</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/spark" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f525.svg" alt="spark"/>
      <div class="bigdata-label">components/spark</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/s3_minio" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e5.svg" alt="s3_minio"/>
      <div class="bigdata-label">components/s3_minio</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/cassandra" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f52c.svg" alt="cassandra"/>
      <div class="bigdata-label">components/cassandra</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/hbase" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" alt="hbase"/>
      <div class="bigdata-label">components/hbase</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/iceberg" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/2744.svg" alt="iceberg"/>
      <div class="bigdata-label">components/iceberg</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/delta_lake" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f30a.svg" alt="delta_lake"/>
      <div class="bigdata-label">components/delta_lake</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/hdfs" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f5c2.svg" alt="hdfs"/>
      <div class="bigdata-label">components/hdfs</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/logstash" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50e.svg" alt="logstash"/>
      <div class="bigdata-label">components/logstash</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/kafka" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4e1.svg" alt="kafka"/>
      <div class="bigdata-label">components/kafka</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/nifi" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50c.svg" alt="nifi"/>
      <div class="bigdata-label">components/nifi</div>
    </a>
    <a class="bigdata-item" href="/#/bigdata/components/README" target="_blank">
      <img class="bigdata-icon" src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4d6.svg" alt="README"/>
      <div class="bigdata-label">components/README</div>
    </a>
  </div>
</div>


















