"""Book card vocabulary v3: genre (derived from a sub-genre), elements, style scales.

Three layers, each answering one kind of question and verified one way
(docs/card-schema-v3.md): the sub-genre says what kind of book it is, elements say
whether a device or setting is present, style scales say how it reads.

v3 (2026-10-10) rebuilt the sub-genre list from the API pilot under one rule: a
sub-genre must belong to exactly one genre (the genre is derived from it) and be
separable from its siblings by a one-line definition. Anything that is really an
element (系统, 穿越, 直播, 同人, 争霸) is not a sub-genre. Absence claims (无感情线,
单女主) are not elements either: one quote cannot prove a negative. Who the book is
about (单一男主 / 单一女主 / 群像) is a style scale, not an element.
"""

from __future__ import annotations

# 一级题材 -> 二级候选。模型只选二级，一级由 SUBGENRE_TO_GENRE 推导。
SUBGENRES: dict[str, tuple[str, ...]] = {
    "玄幻": ("东方玄幻", "异世大陆", "高武世界"),
    "奇幻": ("剑与魔法", "现代魔法", "史诗奇幻"),
    "武侠": ("传统武侠", "武侠幻想", "国术无双"),
    "仙侠": ("古典仙侠", "幻想修仙", "现代修真"),
    "都市": ("都市生活", "都市异能", "青春校园", "娱乐明星", "商战职场", "官场仕途", "人间百态"),
    "历史": ("架空历史", "上古先秦", "秦汉三国", "两晋隋唐", "五代十国", "两宋元明", "清史民国", "外国历史"),
    "军事": ("军旅特战", "抗战烽火", "谍战特工"),
    "科幻": ("未来科技", "星际文明", "进化变异", "末世危机"),
    "游戏": ("电子竞技", "虚拟网游", "游戏异界"),
    "悬疑灵异": ("诡异神秘", "侦探推理", "灵异民俗"),
    "诸天无限": ("无限", "诸天", "综漫"),
    "言情": ("古代言情", "现代言情", "幻想言情"),
}
GENRES: tuple[str, ...] = tuple(SUBGENRES)
SUBGENRE_TO_GENRE: dict[str, str] = {sub: genre for genre, subs in SUBGENRES.items() for sub in subs}
UNKNOWN_GENRE = "其他"

# 一句话边界，只给容易混的二级；写进 prompt。
SUBGENRE_NOTES: dict[str, str] = {
    "东方玄幻": "架空世界，斗气、武魂等非修仙体系；有炼气筑基金丹这套体系的归仙侠",
    "异世大陆": "主角从现实世界来到的架空大陆",
    "高武世界": "现代或近未来社会里武道、超凡公开化",
    "古典仙侠": "剑仙、神话、志怪气质的仙侠，洪荒封神也归这里",
    "幻想修仙": "架空世界里的修仙升级",
    "现代修真": "现代背景的修仙",
    "都市生活": "现代都市，没有超自然能力",
    "都市异能": "现代都市，主角有超能力或医术、相术、风水等玄术",
    "人间百态": "写实的时代叙事、家庭伦理、市井人生",
    "架空历史": "虚构朝代，或没写明朝代的古代；写明真实朝代就选对应朝代",
    "军旅特战": "当代军人、特种兵、兵王、佣兵的成长与作战",
    "未来科技": "未来或近未来背景，科技是主要看点；现代背景的黑科技归都市",
    "进化变异": "基因、进化、变异是主题，不以末世为主",
    "游戏异界": "游戏世界变成真实世界，或带着游戏能力到异界",
    "诡异神秘": "诡异复苏、规则怪谈、克苏鲁式的不可名状",
    "灵异民俗": "鬼怪、道士、民俗禁忌",
    "无限": "一个个独立副本，完成任务",
    "诸天": "穿梭多个已知作品或世界，不是副本制",
    "综漫": "穿梭的是动漫作品",
    "幻想言情": "古代或架空背景之外的幻想设定，感情线是主线",
}

# 旧版二级和模型常写的名字 -> v3 二级；映射不到的保持原样，由 normalise_subgenre 记录。
SUBGENRE_ALIASES: dict[str, str] = {
    "修真文明": "幻想修仙", "神话修真": "幻想修仙", "异术超能": "都市异能",
    "军旅生涯": "军旅特战", "军事战争": "军旅特战", "特种兵": "军旅特战", "佣兵": "军旅特战",
    "超级科技": "未来科技", "未来世界": "未来科技", "古武机甲": "星际文明",
    "规则怪谈": "诡异神秘", "惊悚微恐": "诡异神秘", "仙侠奇缘": "幻想言情", "浪漫青春": "现代言情",
    "时代叙事": "人间百态", "家庭伦理": "人间百态", "古武未来": "武侠幻想", "武侠同人": "传统武侠",
    "神秘幻想": "史诗奇幻", "另类幻想": "史诗奇幻", "游戏主播": "虚拟网游", "官场": "官场仕途", "仕途": "官场仕途",
    "都市": "都市生活", "现代都市": "都市生活", "玄幻": "东方玄幻", "修仙": "幻想修仙", "仙侠": "幻想修仙",
    "末世": "末世危机", "无限流": "无限", "谍战": "谍战特工", "抗战": "抗战烽火", "网游": "虚拟网游", "电竞": "电子竞技",
}

# 元素：多选，封闭。每条是"看到什么就算有"的依据。分组只为 prompt 可读，输出平铺。
ELEMENT_GROUPS: dict[str, dict[str, str]] = {
    "主角来路": {
        "穿越": "主角从另一个世界或时代来到故事世界",
        "重生": "主角带着前世记忆回到过去或重活一次",
        "穿书": "穿越进一本已知的小说、游戏或影视的世界",
        "强者归来": "开篇主角已是强者，回到弱者身份或故地",
    },
    "外挂与体系": {
        "系统": "有面板、任务、奖励、签到等游戏化外挂",
        "随身空间": "随身携带的空间、位面、农场或仓库",
        "无限流": "主角被投放进一个个独立副本或世界完成任务",
        "诸天": "穿梭多个已知作品或世界，但不是副本制",
        "御兽": "以收服、培养宠物、异兽或召唤物为主要战力",
        "炼丹炼器": "丹药、炼器、阵法、符箓等技艺是主角的主要手段",
        "卡牌": "以卡牌、技能卡、抽卡为核心体系",
        "变身": "主角性别或种族发生变化并持续",
    },
    "流派": {
        "升级流": "情节以境界或等级逐级提升为主线",
        "凡人流": "主角资质平庸，靠谨慎、积累、运气缓慢变强",
        "种田经营": "发展领地、家族、店铺、产业是主线",
        "争霸建国": "势力扩张、攻城略地、建国称帝是主线",
        "学院": "校园或学院是主要舞台",
        "技术流": "以现代知识、科技、工业改变世界为主线",
        "商战": "公司、资本、商业竞争是主线",
        "军旅": "当代军队生活或作战是主线",
        "谍战": "情报、潜伏、反间是主线",
        "娱乐圈": "演艺、文娱产业是主线",
        "直播": "直播或主播是主要形式",
        "游戏": "网游、电竞、游戏制作是主线",
        "美食": "烹饪、餐饮是主线",
        "医术": "医生、医术是主线",
        "刑侦推理": "破案、刑侦、法医、推理是主线",
        "盗墓探险": "盗墓、探险、寻宝是主线",
        "年代": "以 1949 至 2000 年代的中国为背景的现实生活",
        "玄学鉴宝": "相术、风水、鉴宝、算命等玄学技艺是主角主要手段",
        "权谋": "朝堂、家族或势力之间的谋略博弈是主要看点",
        "复仇": "主角以报仇为主线动机",
        "扮猪吃虎": "主角长期隐藏实力或身份，反复在被轻视后翻盘",
    },
    "世界设定": {
        "修仙": "修真、炼气、筑基、飞升等体系",
        "高武": "现代或近未来社会里武道或超凡公开化",
        "异能": "现代背景，人物拥有非修炼来源的特殊能力",
        "末世": "文明崩溃后的生存",
        "丧尸": "丧尸或僵尸是主要威胁",
        "机甲星际": "机甲、星舰、星际文明",
        "克苏鲁诡异": "不可名状、理智值、规则怪谈、诡异复苏",
        "鬼怪灵异": "鬼、灵体、诅咒、凶宅等超自然恐怖",
        "同人": "以已有作品为底本（关键词里写原作名）",
    },
    "感情": {
        "后宫": "两名以上异性与主角并存的、被叙事认可的感情或伴侣关系",
    },
}
ELEMENTS: dict[str, str] = {label: definition for group in ELEMENT_GROUPS.values() for label, definition in group.items()}

# 摘录里必须出现的特征词（记 element_weak_quote）。v3 试点上量过：对 系统 / 修仙 / 穿越 / 重生 开门槛拦下 37 条，
# 几乎全是真证据（"他本是地球上的一名普通上班族"、"怒气点：465"、"眼前弹出一个光屏"），v3 的 prompt 已经让摘录
# 带上特征，词面门槛只剩误杀，所以清空；机制留着，以后某个元素的摘录又飘了再按元素加。
ELEMENT_QUOTE_SIGNATURES: dict[str, tuple[str, ...]] = {}

# Elements whose evidence is a list of names rather than a quote. A harem is spread over many scenes
# (six girls in six chapters), no 20-character sentence proves it, and Flash copied the definition
# instead on 10 of 116 pilot cards; two names that both occur in the digest are checkable evidence.
ELEMENT_EVIDENCE_NAMES: dict[str, str] = {
    "后宫": "依据不写摘录，写两位以上与主角有感情或伴侣关系的异性名字，用、分隔，名字必须在档案里出现过",
}

# 模型常写的近义词 / 二级题材名 -> 词表里的元素。
ELEMENT_ALIASES: dict[str, str] = {
    "空间": "随身空间", "随身农场": "随身空间", "系统流": "系统", "金手指": "系统",
    "都市异能": "异能", "异术超能": "异能", "超能力": "异能",
    "星际文明": "机甲星际", "机甲": "机甲星际", "星际": "机甲星际",
    "谍战特工": "谍战", "特工": "谍战", "军旅特战": "军旅", "特种兵": "军旅", "佣兵": "军旅", "兵王": "军旅",
    "商战职场": "商战", "职场": "商战", "创业": "商战",
    "综漫": "诸天", "二次元": "同人", "动漫": "同人",
    "末世危机": "末世", "规则怪谈": "克苏鲁诡异", "诡异": "克苏鲁诡异", "克苏鲁": "克苏鲁诡异",
    "灵异民俗": "鬼怪灵异", "灵异": "鬼怪灵异", "鬼怪": "鬼怪灵异", "中式恐怖": "鬼怪灵异",
    "侦探推理": "刑侦推理", "刑侦": "刑侦推理", "法医": "刑侦推理", "推理": "刑侦推理", "破案": "刑侦推理",
    "女装": "变身", "性转": "变身",
    "争霸": "争霸建国", "王朝争霸": "争霸建国", "建国": "争霸建国",
    "种田": "种田经营", "经营": "种田经营",
    "修真": "修仙", "修仙文明": "修仙",
    "网游": "游戏", "电竞": "游戏", "虚拟网游": "游戏",
    "娱乐明星": "娱乐圈", "文娱": "娱乐圈", "明星": "娱乐圈",
    "校园": "学院", "青春校园": "学院",
    "多女主": "后宫",
    "鉴宝": "玄学鉴宝", "风水": "玄学鉴宝", "玄学": "玄学鉴宝", "相术": "玄学鉴宝",
    "宫斗": "权谋", "朝堂": "权谋", "权谋争斗": "权谋", "官场": "权谋",
    "报仇": "复仇", "复仇打脸": "复仇", "扮猪吃老虎": "扮猪吃虎", "隐藏实力": "扮猪吃虎",
    "重生归来": "重生", "穿越重生": "穿越", "穿越者": "穿越",
    "盗墓": "盗墓探险", "探险": "盗墓探险",
    "美食文": "美食", "医生": "医术",
    "年代文": "年代",
}

# 风格：每个维度单选。
# 节奏 was a scale until card_v2.4: two API models disagreed on it for 55 of 119 books (dropped 2026-10-10).
# 主角结构 replaced the elements 群像 / 女主视角 in v3: who the book follows is one choice, not a presence claim.
STYLE_DIMENSIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "主角结构": (("单一男主", "单一女主", "群像"), "群像：多视角、多主角，没有单一绝对主角"),
    "爽度": (("高", "中", "低"), "高：冲突迅速解决、主角连续占上风、反复打脸；低：长期受挫、代价沉重"),
    "基调": (("轻松搞笑", "热血", "沉重压抑", "温馨日常", "悬疑惊悚"), "取最主要的一种"),
    "感情线比重": (("无", "辅线", "主线"), "主线：情节围绕两人关系推进"),
    "主角起点": (("开局无敌", "普通", "废柴逆袭"), "开篇主角实力相对周围人的位置"),
}
STYLE_OPTIONS: dict[str, tuple[str, ...]] = {dim: options for dim, (options, _) in STYLE_DIMENSIONS.items()}

MAX_KEYWORDS = 5


def genre_of(subgenre: str) -> str:
    return SUBGENRE_TO_GENRE.get(subgenre, UNKNOWN_GENRE)


def vocabulary_text() -> str:
    """The vocabulary as the prompt shows it."""

    lines = ["【题材二级，只能选一个；每行冒号前是一级，不要填一级】"]
    for genre, subs in SUBGENRES.items():
        lines.append(f"{genre}：" + "、".join(subs))
    lines.append("容易混的二级怎么分：")
    lines.extend(f"- {sub}：{note}" for sub, note in SUBGENRE_NOTES.items())
    lines.append("")
    lines.append("【元素，多选，只列出有的；每条后面是算有的依据】")
    for group, items in ELEMENT_GROUPS.items():
        lines.append(f"{group}：")
        lines.extend(f"- {label}：{definition}" for label, definition in items.items())
    lines.append("")
    lines.append("【风格，每项只能选一档】")
    for dim, (options, definition) in STYLE_DIMENSIONS.items():
        lines.append(f"- {dim}：{' / '.join(options)}（{definition}）")
    return "\n".join(lines)
