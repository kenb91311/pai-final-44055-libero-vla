# 第三者実装と公開資産

- RTCのguidance実装は、固定版LeRobotの構造・式を利用し、denoiser入力の微分を有効にする変更を加えています。該当ファイルのcopyright・変更説明と、LICENSES/LeRobot-APACHE-2.0.txtを保持しています。
- A2C2の補正headは作者の固定版から移植しています。該当ファイルの出典・変更説明と、LICENSES/A2C2-UPSTREAM-LICENSE.txtを保持しています。出力を残差と解釈する版と直接行動と解釈する版を区別しています。
- VLASHは作者の固定ソースをSHA検証して取得します。LICENSES/VLASH-APACHE-2.0.txtとsource manifestを添付しています。

コードの条件と、外部のモデル重み・tokenizer・meshの条件は別です。重みは含めていません。各資産の固定revisionと確認範囲はdocs/reference/を参照してください。コード全体や外部資産を一括して同じライセンスと宣言するものではありません。
