<script setup>
import { ref, watch } from "vue";
import { askQuestion, rebuildIndex } from "./api.js";

const KEY_STORAGE = "medical_deepseek_api_key";
const MODEL_STORAGE = "medical_deepseek_model";

const apiKey = ref(localStorage.getItem(KEY_STORAGE) || "");
const model = ref(localStorage.getItem(MODEL_STORAGE) || "deepseek-chat");
const question = ref("");
const answer = ref("这里将展示模型回答。");
const sources = ref([]);
const statusText = ref("就绪");
const loading = ref(false);

watch(apiKey, (value) => localStorage.setItem(KEY_STORAGE, value));
watch(model, (value) => localStorage.setItem(MODEL_STORAGE, value), { immediate: true });

async function handleAsk() {
  const q = question.value.trim();
  const key = apiKey.value.trim();
  if (!key) {
    alert("请填写 DeepSeek API Key");
    return;
  }
  if (!q) {
    alert("请输入问题");
    return;
  }

  loading.value = true;
  statusText.value = "检索与生成中...";
  answer.value = "正在回答...";
  sources.value = [];

  try {
    const data = await askQuestion(q, key, model.value);
    answer.value = data.answer || "";
    sources.value = data.sources || [];
    statusText.value = "完成";
  } catch (err) {
    answer.value = `错误: ${err.message}`;
    statusText.value = "失败";
  } finally {
    loading.value = false;
  }
}

async function handleRebuild() {
  loading.value = true;
  statusText.value = "重建索引中...";
  answer.value = "正在重建索引...";
  sources.value = [];

  try {
    const data = await rebuildIndex();
    answer.value = data.message || "索引构建完成";
    statusText.value = "索引已更新";
  } catch (err) {
    answer.value = `重建失败: ${err.message}`;
    statusText.value = "重建失败";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="layout">
    <main class="workspace">
      <header class="page-header">
        <a class="site-logo" href="/" aria-label="RAG">RAG</a>
        <div class="page-header-text">
          <div class="brand-inline">AI问答</div>
        </div>
        <button
          class="btn btn-secondary"
          :disabled="loading"
          @click="handleRebuild"
        >
          重建索引
        </button>
      </header>

      <section class="card">
        <div class="settings">
          <div>
            <label class="input-label" for="apiKeyInput">DeepSeek API Key</label>
            <input
              id="apiKeyInput"
              class="text-input"
              type="password"
              v-model="apiKey"
              placeholder="只保存在本机浏览器，不会写入服务器"
              autocomplete="off"
            />
          </div>
          <div>
            <label class="input-label" for="modelSelect">模型</label>
            <select id="modelSelect" class="text-input" v-model="model">
              <option value="deepseek-chat">deepseek-chat</option>
              <option value="deepseek-reasoner">deepseek-reasoner</option>
            </select>
          </div>
        </div>
        <label class="input-label" for="questionInput">请输入你的课程问题</label>
        <textarea
          id="questionInput"
          v-model="question"
          placeholder="例如：数据库系统由哪几部分组成？"
        />
        <div class="actions">
          <span class="status-chip">{{ statusText }}</span>
          <button
            class="btn btn-primary"
            :disabled="loading"
            @click="handleAsk"
          >
            提交问题
          </button>
        </div>
      </section>

      <section class="card card-grow">
        <h2>回答结果</h2>
        <pre>{{ answer }}</pre>
      </section>

      <section class="card">
        <h2>检索片段</h2>
        <div v-if="!sources.length" class="source-item">暂无检索片段。</div>
        <div
          v-for="item in sources"
          :key="item.rank"
          class="source-item"
        >
          #{{ item.rank }} (score={{ item.score ?? "N/A" }})
          <template v-if="item.metadata?.chapter || item.metadata?.title">
{{ [item.metadata.chapter, item.metadata.title].filter(Boolean).join(" · ") }}
          </template>
{{ item.text }}
        </div>
      </section>
    </main>
  </div>
</template>
