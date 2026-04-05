import { webview } from '@kit.ArkWeb';

@Entry
@Component
struct Index {
  // 新的公网链接（您提供的）
  private webUrl: string = 'https://intelligent-learning-navigator-gh2hwp5er52wlyvru9trml.streamlit.app/';
  private webController: webview.WebviewController = new webview.WebviewController();

  aboutToAppear() {
    // 可选：设置自定义User-Agent，提升兼容性（避免某些页面要求登录）
    try {
      const customUA = 'Mozilla/5.0 (Linux; Android 12; NOH-AN00) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.196 Mobile Safari/537.36';
      this.webController.setCustomUserAgent(customUA);
    } catch (error) {
      console.error(`setCustomUserAgent error: ${error}`);
    }
  }

  build() {
    Column() {
      Web({ src: this.webUrl, controller: this.webController })
        .width('100%')
        .height('100%')
        .javaScriptAccess(true)
        .domStorageAccess(true)
        .fileAccess(false)
        .mediaPlayGestureAccess(true)
        .multiWindowAccess(false)
        .horizontalScrollBarAccess(false)
        .verticalScrollBarAccess(false)
        .onPageEnd((event) => {
          console.info('Page loaded: ' + event.url);
        })
        .onErrorReceive((event) => {
          console.error('WebView error: ' + event.error.getErrorInfo());
        });
    }
    .width('100%')
    .height('100%')
  }
}