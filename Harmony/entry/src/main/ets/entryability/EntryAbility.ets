import { UIAbility } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';

export default class EntryAbility extends UIAbility {
  onWindowStageCreate(windowStage: window.WindowStage): void {
    // 直接加载主页面（Index.ets），所有WebView配置在Index中完成
    windowStage.loadContent('pages/Index', (err) => {
      if (err) {
        console.error('Failed to load content', err);
        return;
      }
    });
  }
}