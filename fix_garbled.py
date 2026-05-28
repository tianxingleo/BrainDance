"""Fix garbled Chinese text in SparkGaussianViewer.vue caused by UTF-8->GBK double encoding."""

filepath = '3dgs_viewer/spark-3dgs-viewer/src/components/SparkGaussianViewer.vue'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Each tuple: (garbled_text, correct_text)
replacements = [
    # Orbit mode section header
    ('鐩告満妯″紡', '相机模式'),
    # Orbit variable comments
    ('妯″紡锛氱鏈虹虹粫妯″瀷涓績鑷姩鏃嬭浆锛堝渾鍛ㄨ繍鍔級锛屼笉鏀瑰彉鐜版湁鎵嬪姩鎺у埗閫昏缉', '模式：相机绕模型中心自动旋转（圆周运动），不改变现有手动控制逻辑'),
    ('鏄惁寮€鍚?orbit 妯″紡', '是否开启 orbit 模式'),
    ('鏄惁鏆傚仠鏃嬭浆', '是否暂停旋转'),
    ('鏃嬭浆閫熷害锛屽崟浣嶏細搴?绉?', '旋转速度，单位：度/秒'),
    ('鏃嬭浆鏂瑰悜锛?=閫嗘椂閽?CCW)锛?1=椤烘椂閽?CW)', '旋转方向：1=逆时针(CCW)，-1=顺时针(CW)'),
    ('鏃嬭浆鍗婂緞锛?=鑷姩璁＄畻锛堜繚鎸佸綋鍓嶈窛绂伙級锛?0 浣跨敤鎸囧畾鍊?', '旋转半径：0=自动计算（保持当前距离），>0 使用指定值'),
    # Orbit runtime variables
    ('杩愯鏃跺彉閲忥紙闈炲搷搴斿紡锛岄伩鍏嶄笉蹇呰鐨?Vue 閲嶆覆鏌擄級', '运行时变量（非响应式，避免不必要的 Vue 重渲染）'),
    ('褰撳墠鏃嬭浆瑙掑害锛堝姬搴︼級', '当前旋转角度（弧度）'),
    ('鐩告満 Y 鍧愭爣锛堜繚鎸?orbit 楂樺害涓嶅彉锛?', '相机 Y 坐标（保持 orbit 高度不变）'),
    ('鑷姩璁＄畻鐨勮建閬撳崐寰勶紙浠庡綋鍓嶇浉鏈鸿窛绂昏幏鍙栵級', '自动计算的轨道半径（从当前相机距离获取）'),
    ('鐢ㄦ埛姝ｅ湪鎵嬪姩鎷栨嫿锛堜复鏃朵腑鏂?orbit锛?', '用户正在手动拖拽（临时中断 orbit）'),
    # syncOrbitFromCamera comments
    ('浠庡綋鍓嶇浉鏈轰綅缃悓姝?orbit 鍙傛暟锛堣搴︺€侀珮搴︺€佸崐寰勶級', '从当前相机位置同步 orbit 参数（角度、高度、半径）'),
    ('鐢ㄤ簬锛?1) 寮€鍚?orbit 鏃跺垵濮嬪寲 (2) 鎵嬪姩鎷栨嫿鏉惧紑鍚庢仮澶?(3) 妯″瀷鍔犺浇鍚庨噸鏂伴敋瀹?', '用于：(1) 开启 orbit 时初始化 (2) 手动拖拽松开后恢复 (3) 模型加载后重新锚定'),
    # startOrbit/stopOrbit/toggleOrbitPause comments
    ('寮€鍚?orbit 妯″紡锛氫粠褰撳墠鐩告満浣嶇疆鍒濆寲杞ㄩ亾鍙傛暟骞跺紑濮嬫棆杞?', '开启 orbit 模式：从当前位置初始化轨道参数并开始旋转'),
    ('濡傛灉鑷姩鍗婂緞杩囧皬锛堢浉鏈哄お闈犺繎涓績锛夛紝浣跨敤榛樿窛绂', '如果自动半径过小（相机太靠近中心），使用默认距离'),
    ('鍏抽棴 orbit 妯″紡锛氱浉鏈哄仠鐣欏湪褰撳墠浣嶇疆锛屾仮澶嶅師鏈夋墜鍔ㄦ帶鍒?', '关闭 orbit 模式：相机停在当前位置，恢复原有手动控制'),
    ('鏆傚仠/鎭値 orbit 鏃嬭浆', '暂停/恢复 orbit 旋转'),
    ('鎭値鏃朵粠褰撳墠鐩告満浣嶇疆閲嶆柊鍚屾杞ㄩ亾鍙傛暟锛岀‘淇濇棤缂濊鎺?', '恢复时从当前位置重新同步轨道参数，确保无缝衔接'),
    ('鍒囨崲鏃嬭浆鏂瑰悜锛堥『鏃堕拡/閫嗘椂閽堬級', '切换旋转方向（顺时针/逆时针）'),
    # flyToImage comment
    ('搴曢儴闀滃ご浠ｈ〃鐪熷疄閲囬泦鐩告満锛岃烦杞悗蹇呴』鎸夌涓€浜虹О鐩告満缁х画浜や簰銆?', '底部镜头代表真实采集相机，跳转后必须按第一人称相机继续交互。'),
    # highlightStatus values
    ('楂樹寒闀滃ご:', '高亮镜头:'),
    ('楂樹寒褰撳墠瑙嗚鍖哄煙', '高亮当前视角区域'),
    # alert message
    ('鍦烘櫙涓病鏈夋壘鍒扮鍚堣鎻忚堪鐨勮嗚鍟?', '场景中没有找到符合该描述的视角哦~'),
    # console.error
    ('鍔犺浇浣嶅Э澶辫触:', '加载位姿失败:'),
    # pointer event comments
    ('Orbit 鎸囬拡浜嬩欢锛氭娴嬫墜鍔ㄦ嫋鎷戒互涓存椂涓柆/鎭値 orbit 鏃嬭浆', 'Orbit 指针事件：检测手动拖拽以临时中断/恢复 orbit 旋转'),
    ('鎸変笅鏃舵爣璁版嫋鎷界姸鎬侊紝鏆傚仠 orbit', '按下时标记拖拽状态，暂停 orbit'),
    ('鏉炬墜鍚庝粠褰撳墠鐩告満浣嶇疆閲嶆柊鍚屾杞ㄩ亾鍙傛暟锛屽疄鐜版棤缂濇仮澶?', '松手后从当前位置重新同步轨道参数，实现无缝恢复'),
    ('pointerleave 浣滀负瀹夊叏鍏滃簳锛岄槻姝㈡嫋鎷界姸鎬佸崱浣?', 'pointerleave 作为安全兜底，防止拖拽状态卡住'),
    # animation loop comments
    ('Orbit 妯″紡锛氳嚜鍔ㄦ棆杞浉鏈猴紙闈炴殏鍋殏鍋溿€侀潪鎵嬪姩鎷栨嫿鏃剁敓鏁堬級', 'Orbit 模式：自动旋转相机（非暂停、非手动拖拽时生效）'),
    ('鏍规嵁閫熷害鍜屾柟鍚戞洿鏂版棆杞搴?', '根据速度和方向更新旋转角度'),
    ('璁＄畻鍦嗗懆浣嶇疆骞跺簲鐢ㄥ埌鐩告満', '计算圆周位置并应用到相机'),
    ('濮嬬粓鏇存柊 controls锛堝鐞嗘粴杞缉鏀俱€佹墜鍔ㄦ嫋鎷界瓑杈撳叆锛?', '始终更新 controls（处理滚轮缩放、手动拖拽等输入）'),
    ('Orbit 妯″紡涓嬮噸鏂伴攣瀹氱浉鏈轰綅缃紝闃柊 controls 瑕嗙洊 orbit 浣嶇疆', 'Orbit 模式下重新锁定相机位置，防止 controls 覆盖 orbit 位置'),
    # sceneCenter sync
    ('濡傛灉 orbit 宸插紑鍚紝閲嶆柊鍚屾杞ㄩ亾鍙傛暟鍒版柊璁＄畻鐨勬鍨嬩腑蹇?', '如果 orbit 已开启，重新同步轨道参数到新计算的模型中心'),
    # notifyFlutter
    ('Spark 妯″瀷鍔犺浇瀹屾垚', 'Spark 模型加载完成'),
    # clip offset
    ('鍓栧垏浣嶇疆:', '剖切位置:'),
    # onMounted comments
    ('鏀跺埌鍔犺浇璇锋眰:', '收到加载请求:'),
    ('鏀跺埌 TimePeeling 妯″瀷鍒楄〃:', '收到 TimePeeling 模型列表:'),
    ('褰撳墠妯″瀷:', '当前模型:'),
    ('Spark 2.0 褰撳墠鐗堟湰鏆備笉鏀寔 TimePeeling 鍒囨崲锛屼絾闇€佹彁渚涚┖瀹炵幇閬垮厤 Flutter 绔鎶ラ敊', 'Spark 2.0 当前版本暂不支持 TimePeeling 切换，但需要提供空实现避免 Flutter 端报错'),
    ('鍚庣画鍙墿灞曚负澶氭ā鍨嬪垏鎹㈤€昏缉', '后续可扩展为多模型切换逻辑'),
    # Quality HUD labels
    ('娴佺晠', '流畅'),
    ('鏍囧噯', '标准'),
    ('楂樻竻', '高清'),
    ('鍥炲繂', '回忆'),
    ('瑙傚療', '观察'),
    ('鑷敱', '自由'),
    ('璺緞', '路径'),
    ('鏆傚仠', '暂停'),
    ('缁х画', '继续'),
    ('鍋滄', '停止'),
    # Orbit panel comment
    ('Orbit 鎺у埗闈㈡澘锛氱鏈虹粫妯″瀷涓績鑷姩鏃嬭浆', 'Orbit 控制面板：相机绕模型中心自动旋转'),
    ('杞ㄩ亾鏃嬭浆', '轨道旋转'),
    ('鎭値', '恢复'),
    ('鏃嬭浆閫熷害鎺у埗', '旋转速度控制'),
    ('鏃嬭浆鍗婂緞鎺у埗', '旋转半径控制'),
    ('鏃嬭浆鏂瑰悜鍒囨崲', '旋转方向切换'),
    # interaction mode
    ('浜や簰妯″紡:', '交互模式:'),
    # module title comment
    ('妯″紡', '模式'),
]

count = 0
for garbled, correct in replacements:
    if garbled in content:
        occurrences = content.count(garbled)
        content = content.replace(garbled, correct)
        count += occurrences
        print(f'  Fixed: "{garbled[:30]}..." -> "{correct[:30]}..." ({occurrences}x)')
    else:
        print(f'  Skipped (not found): "{garbled[:30]}..."')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\nTotal replacements: {count}')
print('Done')
