ps aux | grep -E '[u]vicorn|[f]rontend' | awk '{print $2}' | xargs -r kill -9
