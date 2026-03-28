import { webview } from '@kit.ArkWeb';

@Entry
@Component
struct Index {
  // 确认链接末尾正确: agiqew
  private url: string = 'https://intelligent-learning-navigator-k7khprbwozg22ny4agiqew.streamlit.app';
  private webController: webview.WebviewController = new webview.WebviewController();

  aboutToAppear() {
    // 可选：设置自定义 User-Agent，提升兼容性（方法替代已废弃的 .userAgent() 属性）
    try {
      // 使用标准 Android WebView UA，让 Streamlit 返回移动端界面
      const customUA = 'Mozilla/5.0 (Linux; Android 12; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36';
      this.webController.setCustomUserAgent(customUA);
    } catch (error) {
      console.error(`setCustomUserAgent error: ${error}`);
    }
  }

  build() {
    Column() {
      Web({ src: this.url, controller: this.webController })
        .width('100%')
        .height('100%')
        .javaScriptAccess(true)          // ✅ 启用 JavaScript
        .domStorageAccess(true)          // ✅ 启用 DOM 存储
        .fileAccess(false)               // ✅ 禁止本地文件访问
        .mediaPlayGestureAccess(true)    // ✅ 允许媒体自动播放（若需要）
        .multiWindowAccess(false)        // ✅ 禁用多窗口
        .horizontalScrollBarAccess(false)
        .verticalScrollBarAccess(false)
      // ⚠️ 移除 .mixedMode() 和 .cacheMode()
      // 原因：在当前 API 12 版本中，直接使用可能引发编译错误，
      // 且 Streamlit 应用在默认的混合内容与缓存策略下通常可正常工作。
    }
    .width('100%')
    .height('100%')
  }
}