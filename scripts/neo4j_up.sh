#!/usr/bin/env bash
# Bring up the Neo4j the notebooks expect, by whichever route is available.
#
#   bolt://localhost:7690   browser http://localhost:7476   neo4j / sandbox-kg
#
# Route 1 (preferred): Docker. Route 2: a native tarball under ./.neo4j.
# Either way, `scripts/neo4j_restore.py` reloads notebook 02's graph afterwards.
set -euo pipefail

NAME=extraction-sandbox-neo4j
BOLT=7690
HTTP=7476
PASS=sandbox-kg
VERSION=5.26.19
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL="$HERE/.neo4j/neo4j-community-$VERSION"

reachable() { nc -z -G 2 localhost "$BOLT" >/dev/null 2>&1; }

if reachable; then
  echo "neo4j already reachable on bolt://localhost:$BOLT"
  exit 0
fi

if docker info >/dev/null 2>&1; then
  echo "docker is up — using a container"
  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker start "$NAME" >/dev/null
  else
    docker run -d --name "$NAME" -p "$BOLT":7687 -p "$HTTP":7474 \
      -e NEO4J_AUTH="neo4j/$PASS" -e NEO4J_PLUGINS='["apoc"]' \
      -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
      "neo4j:${VERSION%.*}" >/dev/null
  fi
else
  echo "docker is unavailable — falling back to a local install at $LOCAL"
  if [ ! -d "$LOCAL" ]; then
    mkdir -p "$HERE/.neo4j"
    curl -fsSL "https://dist.neo4j.org/neo4j-community-$VERSION-unix.tar.gz" \
      | tar xz -C "$HERE/.neo4j"
    APOC="https://github.com/neo4j/apoc/releases/download/$VERSION/apoc-$VERSION-core.jar"
    curl -fsSL "$APOC" -o "$LOCAL/plugins/apoc.jar" || \
      echo "  (apoc download failed; the loader falls back to method='grouped')"
    { echo "server.bolt.listen_address=:$BOLT"
      echo "server.http.listen_address=:$HTTP"
      echo "dbms.security.procedures.unrestricted=apoc.*"; } >> "$LOCAL/conf/neo4j.conf"
    JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home -v 21 2>/dev/null || echo /opt/homebrew/opt/openjdk@21)}" \
      "$LOCAL/bin/neo4j-admin" dbms set-initial-password "$PASS"
  fi
  JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home -v 21 2>/dev/null || echo /opt/homebrew/opt/openjdk@21)}" \
    "$LOCAL/bin/neo4j" start
fi

printf "waiting for bolt"
for _ in $(seq 60); do reachable && { echo " — up"; exit 0; }; printf .; sleep 2; done
echo; echo "timed out waiting for bolt://localhost:$BOLT" >&2; exit 1
