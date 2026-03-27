#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUPABASE_DIR="$ROOT_DIR/supabase"

function bd_read_status_value() {
  local key="$1"
  supabase status -o env | sed -n "s/^${key}=\"\\(.*\\)\"/\\1/p" | head -n1
}

function bd_api_url() {
  bd_read_status_value "API_URL"
}

function bd_db_url() {
  bd_read_status_value "DB_URL"
}

function bd_service_role_key() {
  bd_read_status_value "SERVICE_ROLE_KEY"
}

function bd_anon_key() {
  bd_read_status_value "ANON_KEY"
}

function bd_psql() {
  local db_url
  db_url="${DB_URL_OVERRIDE:-$(bd_db_url)}"
  if [[ -z "$db_url" ]]; then
    echo "[bd-common] missing DB_URL" >&2
    return 1
  fi
  psql "$db_url" "$@"
}

function bd_require_local_supabase() {
  if ! bd_psql -c "select 1;" >/dev/null 2>&1; then
    echo "[bd-common] local supabase db is not reachable" >&2
    return 1
  fi
}

function bd_upload_storage_object() {
  local bucket="$1"
  local object_path="$2"
  local local_file="$3"
  local content_type="$4"
  local api_url
  local service_key
  api_url="$(bd_api_url)"
  service_key="$(bd_service_role_key)"
  if [[ -z "$api_url" || -z "$service_key" ]]; then
    echo "[bd-common] missing API_URL or SERVICE_ROLE_KEY" >&2
    return 1
  fi
  curl -sS -X POST "${api_url}/storage/v1/object/${bucket}/${object_path}" \
    -H "Authorization: Bearer ${service_key}" \
    -H "apikey: ${service_key}" \
    -H "x-upsert: true" \
    -H "Content-Type: ${content_type}" \
    --data-binary @"${local_file}" >/dev/null
}

function bd_delete_storage_prefix() {
  local bucket="$1"
  local prefix="$2"
  local api_url
  local service_key
  api_url="$(bd_api_url)"
  service_key="$(bd_service_role_key)"
  if [[ -z "$api_url" || -z "$service_key" ]]; then
    echo "[bd-common] missing API_URL or SERVICE_ROLE_KEY" >&2
    return 1
  fi

  local names=()
  while IFS= read -r name; do
    [[ -n "$name" ]] && names+=("$name")
  done < <(bd_psql -At -c "select name from storage.objects where bucket_id='${bucket}' and name like '${prefix}%';")

  for name in "${names[@]}"; do
    curl -sS -X DELETE "${api_url}/storage/v1/object/${bucket}/${name}" \
      -H "Authorization: Bearer ${service_key}" \
      -H "apikey: ${service_key}" >/dev/null || true
  done
}
