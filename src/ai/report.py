"""Generate an evidence-grounded report with the configured AI provider."""

import json
from typing import Any

from src.ai.base import AIProvider
from src.ai.factory import create_provider
from src.core.exceptions import ConfigurationError

INSTRUCTIONS = (
    "あなたは経済データ分析レポートの作成者です。入力された集計値だけを根拠に、"
    "日本語で簡潔に記述してください。データから読み取れる事実、解釈、予測上の"
    "限界を区別してください。因果関係を断定せず、入力にない数値や出来事を"
    "追加しないでください。基準予測は公式予測ではないと明記してください。"
    "予測精度に触れるときは、入力に含まれる評価条件（何か月先の予測を、"
    "何件・どの方式で評価したか）を必ず明記してください。"
    "1か月先予測の評価結果を、12か月先予測の精度として述べてはいけません。"
    "評価条件が入力にない指標は、精度として断定しないでください。"
    "予測値を示すときは、予測の幅（lower〜upper）とその確率の水準を併記し、"
    "幅が過去の誤差にもとづく目安であることを明記してください。"
    "モデル間の差について、検定のp値が0.05以上の場合は差が明確とは言えないと記述してください。"
    "読み手は統計の専門家ではない社内の職員です。専門用語（MAE、p値、予測区間など）は"
    "初出で短く言い換えてください。"
    "Markdownで、次の見出しの順に書いてください: "
    "「## 要点」（3〜5行の箇条書き）、「## 現状」、「## 今後12か月の予測と幅」、"
    "「## 予測の当たり具合」（モデルの比較と検定、予測の幅の当たり具合）、"
    "「## 使ううえでの注意」。"
    "全体で3,000字程度に収め、表は各見出しに1つまでにしてください。"
)

#: Upper limit of the report length in tokens. Japanese text uses roughly one
#: token or more per character, and 4,000 cut a full report short (2026-10-29).
MAX_TOKENS = 8000


def generate_report(context: dict[str, Any], provider: AIProvider | None = None) -> str:
    """Generate a Japanese report through the selected provider.

    ``provider`` defaults to the one chosen by ``AI_PROVIDER`` in .env. This
    function is separate from the regular data pipeline. API usage and
    possible charges occur only when the dedicated report command is invoked.
    """
    provider = provider if provider is not None else create_provider()
    try:
        payload = json.dumps(context, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("AI report context is not valid JSON data.") from exc
    return provider.generate(INSTRUCTIONS, payload, max_tokens=MAX_TOKENS)
