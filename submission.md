estimator.py から提出するコマンドはこちらです。

 ```bash
uv run whest submit --estimator estimator.py
 ```

 生成済みのアーカイブを提出する場合はこちらです。

 ```bash
uv run whest submit submission.tar.gz
 ```

 どちらも実際に送信するコマンドです。送信せず内容を確認するだけなら：

 ```bash
uv run whest submit --estimator estimator.py --dry-run --format plain
