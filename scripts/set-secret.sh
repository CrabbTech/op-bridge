#!/bin/bash
# Store a secret for op-bridge without it ever appearing in a chat or a shell history line.
# Usage: bash scripts/set-secret.sh TYPESAFE_API_KEY
set -e
NAME="${1:?usage: set-secret.sh NAME}"
DIR="$HOME/Music/op-bridge"; FILE="$DIR/secrets.env"
mkdir -p "$DIR"; chmod 700 "$DIR"
printf '%s (input hidden): ' "$NAME"
read -rs VALUE; echo
[ -n "$VALUE" ] || { echo "nothing entered"; exit 1; }
touch "$FILE"; chmod 600 "$FILE"
grep -v "^${NAME}=" "$FILE" > "$FILE.tmp" || true
printf '%s=%s\n' "$NAME" "$VALUE" >> "$FILE.tmp"
mv "$FILE.tmp" "$FILE"; chmod 600 "$FILE"
unset VALUE
echo "saved $NAME to $FILE"
