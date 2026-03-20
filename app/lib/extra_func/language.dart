enum Language {
  zh(
    name: "简体中文",
    map: {
      "lang": "简体中文",
      "locale": "zh_CN",
      "title": "流光 · 记",
      "home_page": "流光 · 记",
      "pick_null": "请选择",
      "set_lang": "当前语言（点击切换）",
      "set_night": "夜间模式",
      "set_ver": "版本号",
      "set_pub": "发布日期",
      "set_cache": "清除缓存",
      "set_tab1": "显示",
      "set_tab2": "视频",
      "set_tab3": "应用",
      "set_tab4": "帮助",
      "gen_top": "生成自",
      "gen_pic": "图片",
      "gen_text": "文本",
      "gen_video": "视频",
      "gen_tip_pic": "请选择图片文件 (小于4MB)",
      "gen_tip_text": "请输入描述文本",
      "gen_tip_video": "请选择视频文件\n(清晰度小于等于 1080p 30fps，\n拍摄时长 3min 以内)",
      "gen_tip_textbox": "在此处键入文本",
      "gen_button": "生成记忆",
      "gen_shot": "拍照",
      "gen_gallery": "从相册中选择",
      "gen_cancel": "取消",
      "reco_tip_title1": "一、设备与设置",
      "reco_tip_title2": "二、拍摄环境与操作",
      "reco_tip_title3": "三、后期检查",
      "reco_tip1": """设备：使用支持1080p/30fps录像的智能手机或相机。
稳定：建议使用三脚架，避免画面抖动。
分辨率与帧率：在设置中手动选择 1920×1080（1080p） 和 30fps。
曝光：使用自动曝光模式。如光线明暗变化大，可手动锁定曝光。
对焦：使用自动连续对焦模式。""",
      "reco_tip2": """光线：在光线均匀、充足的条件下拍摄。避免逆光（如背景是窗户）。
动作：拍摄时，尽量避免快速移动或晃动相机。
内容：规划不同的拍摄内容和角度，避免画面重复。""",
      "reco_tip3": """如视频使用手机自带相机软件拍摄，请用剪辑软件（如剪映）检查视频，剪掉抖动、模糊或过曝的片段。
最终导出时，确认参数为 1080p 和 30fps。""",
      "reco_camun": "相机访问被拒绝。\n请在系统设置中启用相机权限。",
      "reco_wait": "相机初始化中...",
      "recall": "回忆",
      "record": "相机记录",
      "generate": "图文生成",
      "community": "社区",
      "settings": "设置",
      "manage": "管理",
      "tip_unava": "该功能尚未开放，敬请期待！",
      "tip_oversize": "上传的文件太大了！",
      "tip_overquan": "你上传的文件太多了！",
      "tip_cache": "已成功清除缓存",
      "tip_fail": "文件上传失败！",
      "tip_no_permission": "应用没有权限，无法保存到图库。",
      "recall_search_hint": "搜索回忆...",
      "recall_empty_title": "暂无回忆，去记录一些美好瞬间吧",
      "recall_open_demo": "打开本地离线 Demo 模型",
      "recall_no_desc": "没有描述信息",
      "recall_demo_title": "本地 Demo 模型 (离线可用)",
      "recall_demo_desc": "预置的 3DGS 模型，无需网络即可查看。",
      "recall_error_offline": "加载模型失败，已切换至离线模式",
      "recall_error_search": "搜索失败: ",
      "recall_refresh": "刷新",
      "recall_local_rag": "端侧隐私检索",
      "recall_cloud_rag": "云端语义检索",
      "recall_search_mode": "检索模式",
      "recall_local_ready": "索引已就绪",
      "recall_local_indexing": "正在本地建立向量索引…",
      "recall_local_scope": "搜索词不会上传到云端",
      "recall_local_empty": "本地索引中没有命中结果",
      "recall_cloud_scope": "搜索词将发送到云端大模型",
      "recall_cloud_empty": "云端语义检索没有命中结果",
      // 任务列表页面
      "task_list_title": "记忆任务",
      "task_list_empty_title": "暂无任务",
      "task_list_empty_desc": "开始记录您的第一个记忆吧！",
      "status_pending": "等待处理",
      "status_processing": "处理中",
      "status_completed": "已完成",
      "status_failed": "处理失败",
      "error_not_logged_in": "请先登录",
      "error_fetch_tasks": "获取任务失败",
      "error_title": "发生错误",
      "error_unknown": "未知错误",
      "retry": "重试",
      "refresh": "刷新",
      "task_status_pending": "任务等待处理中...",
      "task_status_processing": "任务正在处理中...",
      "task_status_completed": "任务已完成",
      "task_status_failed": "任务处理失败",
      "task_status_unknown": "任务状态未知",
      // 任务通知弹窗
      "task_completed": "个任务已完成",
      "task_failed": "个任务失败",
      "task_notification_completed": "个记忆任务已完成，点击查看",
      "task_notification_failed": "个记忆任务处理失败，点击查看",
      "logs_collapse": "收起日志",
      "logs_expand": "展开日志",
      "create": "创作",
      "create_guide_subtitle": "选择一种方式开始创建新的记忆。",
      "create_record_desc": "使用相机拍摄并记录空间场景。",
      "create_generate_desc": "通过图片、文本或视频生成记忆模型。",
      "timepeeling": "时间切片",
      "coming_soon": "即将推出",
    },
  ),
  en(
    name: "English",
    map: {
      "lang": "English",
      "locale": "en_US",
      "title": "Brain Dance",
      "home_page": "Brain Dance",
      "pick_null": "Please select",
      "set_lang": "Language (Tap to switch)",
      "set_night": "Night Mode",
      "set_ver": "Version",
      "set_pub": "Publish Date",
      "set_cache": "Clear Cache",
      "set_tab1": "Display",
      "set_tab2": "Video",
      "set_tab3": "App",
      "set_tab4": "Help",
      "gen_top": "Generate from",
      "gen_pic": "Image",
      "gen_text": "Text",
      "gen_video": "Video",
      "gen_tip_pic": "Please select an image file (less than 4MB)",
      "gen_tip_text": "Please enter a description text",
      "gen_tip_video":
          "Please select a video file\n(Resolution ≤ 1080p 30fps,\nDuration within 3 minutes)",
      "gen_tip_textbox": "Type text here",
      "gen_button": "Generate Memory",
      "gen_shot": "Use Camera",
      "gen_gallery": "Choose from Gallery",
      "gen_cancel": "Cancel",
      "reco_tip_title1": "1. Equipment & Settings",
      "reco_tip_title2": "2. Shooting Environment & Operation",
      "reco_tip_title3": "3. Post-Shooting Check",
      "reco_tip1":
          """Device: Use a smartphone or camera that supports 1080p/30fps video recording.
Stability: It is recommended to use a tripod to avoid shaky footage.
Resolution & Frame Rate: Manually select 1920x1080 (1080p) and 30fps in the settings.
Exposure: Use auto-exposure mode. If there are significant changes in lighting, manually lock the exposure.
Focus: Use continuous auto-focus mode.""",
      "reco_tip2":
          """Lighting: Shoot under even and sufficient lighting. Avoid backlighting (e.g., having a window as the background).
Movement: During shooting, try to avoid rapid movements or shaking of the camera.
Content: Plan diverse shooting content and angles to avoid repetitive footage.""",
      "reco_tip3":
          """If the video was shot using a phone's built-in camera app, use editing software (e.g., CapCut) to review the video and trim any shaky, blurry, or overexposed segments.
When finally exporting, confirm the settings are 1080p and 30fps.""",
      "reco_camun":
          "Cannot access the camera.\nPlease check and grant the camera permission in your device settings to continue.",
      "reco_wait": "Initializing the camera...",
      "recall": "Recall",
      "record": "Record",
      "generate": "Generate",
      "community": "Community",
      "settings": "Settings",
      "manage": "Manage",
      "tip_unava": "This feature is not yet available, stay tuned!",
      "tip_oversize": "Uploaded file(s) are too large!",
      "tip_overquan": "You have uploaded too many files!",
      "tip_cache": "The cache is successfully cleared",
      "tip_fail": "Failed to upload file(s)!",
      "tip_no_permission": "Does not have permission to save to the gallery.",
      "recall_search_hint": "Search memories...",
      "recall_empty_title":
          "No memories yet, go record some beautiful moments!",
      "recall_open_demo": "Open local offline Demo model",
      "recall_no_desc": "No description available",
      "recall_demo_title": "Local Demo Model (Available Offline)",
      "recall_demo_desc":
          "Pre-installed 3DGS model, viewable without network connectivity.",
      "recall_error_offline":
          "Failed to load models, switched to offline mode.",
      "recall_error_search": "Search failed: ",
      "recall_refresh": "Refresh",
      "recall_local_rag": "On-device private search",
      "recall_cloud_rag": "Cloud semantic search",
      "recall_search_mode": "Search mode",
      "recall_local_ready": "Local index ready",
      "recall_local_indexing": "Building on-device vector index...",
      "recall_local_scope": "Search terms stay on this device",
      "recall_local_empty": "No local matches found in the memory index",
      "recall_cloud_scope": "Search terms will be sent to the cloud model",
      "recall_cloud_empty": "No results returned from cloud semantic search",
      // Task list page
      "task_list_title": "Memory Tasks",
      "task_list_empty_title": "No Tasks",
      "task_list_empty_desc": "Start recording your first memory!",
      "status_pending": "Pending",
      "status_processing": "Processing",
      "status_completed": "Completed",
      "status_failed": "Failed",
      "error_not_logged_in": "Please log in first",
      "error_fetch_tasks": "Failed to fetch tasks",
      "error_title": "Error Occurred",
      "error_unknown": "Unknown error",
      "retry": "Retry",
      "refresh": "Refresh",
      "task_status_pending": "Task is pending...",
      "task_status_processing": "Task is processing...",
      "task_status_completed": "Task completed",
      "task_status_failed": "Task failed",
      "task_status_unknown": "Unknown task status",
      // Task notification popup
      "task_completed": "task(s) completed",
      "task_failed": "task(s) failed",
      "task_notification_completed": "memory task(s) completed, tap to view",
      "task_notification_failed": "memory task(s) failed, tap to view",
      "logs_collapse": "Collapse logs",
      "logs_expand": "Expand logs",
      "create": "Create",
      "create_guide_subtitle": "Choose a way to create a new memory.",
      "create_record_desc": "Use the camera to capture and record spatial scenes.",
      "create_generate_desc": "Generate memory models from images, text, or video.",
      "timepeeling": "Time Peeling",
      "coming_soon": "Coming soon",
    },
  );

  const Language({required this.name, required this.map});
  final String name;
  final Map<String, String> map;
}

class Localize {
  static Map<String, String> getLangMap(String localeCode) {
    for (var lang in Language.values) {
      if (lang.map['locale'] == localeCode) {
        return lang.map;
      }
    }
    return Language.en.map; // Default to English if not found
  }
}
