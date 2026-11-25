# Mato さんへ：starryai に `/ald/` を追加したいお願い

AutoLineDigitizer を Streamlit で `http://starryai.nims.go.jp/ald/` から使えるようにしたく、nginx 設定の変更をお願いしたいです。構成は **polymer-tool と同じパターン**（tanaka ユーザーサービス → snippet include）で、影響範囲は `/ald/` パスのみです。

## やってほしいこと（1 分作業）

```bash
# 1) snippet を所定の場所にコピー
sudo cp /data/users/tanaka/ald/deploy/ald.conf /etc/nginx/snippets/ald.conf

# 2) 既存の wpd-auto server block に include を 1 行追加
#    場所：include snippets/polymer-tool.conf の近くでお願いします
sudo sed -i '/include snippets\/polymer-tool\.conf/a\    include snippets/ald.conf ;' /etc/nginx/sites-enabled/wpd-auto

# 3) テストしてリロード
sudo nginx -t && sudo systemctl reload nginx

# 4) 確認
curl -s -o /dev/null -w "ald: %{http_code}\n" http://127.0.0.1/ald/
curl -s -o /dev/null -w "wpd-auto: %{http_code}\n" http://127.0.0.1/wpd-auto/
```

`ald: 200` と `wpd-auto: 200` が出れば成功です。

## 追加される snippet の中身

`/data/users/tanaka/ald/deploy/ald.conf`（世界読み可、その場で `cat` で確認可能）：

```nginx
location /ald/ {
    proxy_pass http://127.0.0.1:8766/ald/ ;
    proxy_http_version 1.1 ;
    proxy_set_header Host $host ;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for ;
    proxy_set_header X-Forwarded-Proto $scheme ;
    proxy_set_header X-Real-IP $remote_addr ;
    proxy_set_header Upgrade $http_upgrade ;
    proxy_set_header Connection "upgrade" ;
    client_max_body_size 100M ;
    proxy_read_timeout 600s ;
    proxy_send_timeout 600s ;
    proxy_buffering off ;
}
location = /ald {
    return 301 /ald/ ;
}
```

Streamlit が WebSocket で UI 更新を送るので `proxy_http_version 1.1 + Upgrade/Connection` が必須、PDF アップロードがあるので `client_max_body_size 100M` にしてあります。

## バックエンドと影響範囲

- バックエンド：`127.0.0.1:8766`（tanaka の user systemd service `ald.service`、polymer-tool と同じ流儀）
- 影響するパス：**`/ald/` のみ新規追加**
- `/wpd-auto/` `/polymer-tool/` `/wpd-auto-staging/` 等既存パスは**一切変更なし**

## 戻し方（何かあったら）

```bash
# include 行を 1 行コメントアウト、または削除
sudo sed -i '/include snippets\/ald\.conf/d' /etc/nginx/sites-enabled/wpd-auto
sudo nginx -t && sudo systemctl reload nginx
```

snippet ファイル `/etc/nginx/snippets/ald.conf` は残しても消しても OK です。

## 背景

- 用途：論文 PDF から図を抽出 → wpd-auto に渡す上流ツール（LineFormer 等の重い処理は無し、Streamlit だけ）
- 当面のユーザー：田中とラボ数名、動作見てから広げる想定
- サーバ保存機能は wpd-auto 側にお任せ、ALD 側は図を抽出して橋渡しだけ

よろしくお願いします。
