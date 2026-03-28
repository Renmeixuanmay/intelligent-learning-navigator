import { UIAbility } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import { webview } from '@kit.ArkWeb';

export default class EntryAbility extends UIAbility {
  // 您的Streamlit应用公网链接（替换为您的实际链接）
  private WEB_URL: string = 'https://intelligent-learning-navigator-k7khprbwozg22ny4agigew.streamlit.app';

  onWindowStageCreate(windowStage: window.WindowStage): void {
    // 创建WebView控制器
    let webController: webview.WebviewController = new webview.WebviewController();

    windowStage.loadContent('pages/Index', (err) => {
      if (err) {
        console.error('Failed to load content', err);
        return;
      }

      // 获取窗口并加载WebView
      let windowClass = windowStage.getMainWindowSync();
      windowClass.setUIContent({
        component: (() => {
          // 创建WebView组件并加载URL
          return webview.createWebView({
            src: this.WEB_URL,
            controller: webController
          });
        })()
      });
    });
  }
}