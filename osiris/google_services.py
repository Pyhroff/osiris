"""Gmail + Google Calendar access via OAuth.

One-time setup (same Google Cloud project as your Gemini key):
  1. console.cloud.google.com → APIs & Services → enable "Gmail API" and "Google Calendar API"
  2. Credentials → Create OAuth client ID → Desktop app → download JSON
  3. Save it as `credentials.json` in the OSIRIS project root
First call opens a browser to authorize; the token is cached in token.json after that.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta
from email.mime.text import MIMEText

from . import config

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/drive.readonly",
]
# NOTE: if you authorized before Drive was added, delete token.json and re-run
# — the cached token doesn't carry the new scope.
CREDENTIALS_FILE = config.ROOT / "credentials.json"
TOKEN_FILE = config.ROOT / "token.json"

_services: dict = {}


def _get_service(name: str, version: str):
    """Lazy, cached service builder. Runs the browser OAuth flow on first use."""
    if name in _services:
        return _services[name]

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise RuntimeError(
                    "credentials.json missing — enable Gmail + Calendar APIs in "
                    "console.cloud.google.com, create an OAuth Desktop client, and save "
                    "the downloaded JSON as credentials.json in the OSIRIS folder.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    _services[name] = build(name, version, credentials=creds)
    return _services[name]


# -- Gmail ---------------------------------------------------------------

def gmail_search(query: str, max_results: int = 5) -> list[dict]:
    svc = _get_service("gmail", "v1")
    resp = svc.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    out = []
    for ref in resp.get("messages", []):
        msg = svc.users().messages().get(
            userId="me", id=ref["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"]).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        out.append({"id": ref["id"], "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""), "date": headers.get("Date", ""),
                    "snippet": msg.get("snippet", "")})
    return out


def gmail_read(message_id: str) -> str:
    svc = _get_service("gmail", "v1")
    msg = svc.users().messages().get(userId="me", id=message_id, format="full").execute()

    def walk(part) -> str:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", "replace")
        return "".join(walk(p) for p in part.get("parts", []))

    body = walk(msg["payload"])
    return body.strip() or msg.get("snippet", "(no readable text body)")


def gmail_send(to: str, subject: str, body: str) -> str:
    svc = _get_service("gmail", "v1")
    mime = MIMEText(body)
    mime["to"], mime["subject"] = to, subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    return f"sent (message id {sent['id']})"


def gmail_create_draft(to: str, subject: str, body: str) -> str:
    """Saves to the Drafts folder — nothing is sent until the user hits send."""
    svc = _get_service("gmail", "v1")
    mime = MIMEText(body)
    mime["to"], mime["subject"] = to, subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    d = svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
    return f"draft saved: '{subject}' -> {to} (draft id {d['id']})"


def gmail_reply(message_id: str, body: str) -> str:
    """Draft a reply in the same thread. Nothing sends until the user hits send."""
    svc = _get_service("gmail", "v1")
    msg = svc.users().messages().get(
        userId="me", id=message_id, format="metadata",
        metadataHeaders=["Subject", "From", "Message-ID"]).execute()
    hdrs = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    subject = hdrs.get("Subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    mime = MIMEText(body)
    mime["to"] = hdrs.get("From", "")
    mime["subject"] = subject
    mime["In-Reply-To"] = hdrs.get("Message-ID", "")
    mime["References"] = hdrs.get("Message-ID", "")
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    d = svc.users().drafts().create(
        userId="me",
        body={"message": {"raw": raw, "threadId": msg["threadId"]}}).execute()
    return f"reply draft saved: '{subject}' (draft id {d['id']})"


# -- Drive ---------------------------------------------------------------

_EXPORTABLE = {  # Google-native formats need export, not download
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


def drive_search(query: str, max_results: int = 10) -> list[dict]:
    svc = _get_service("drive", "v3")
    q = query.replace("'", r"\'")
    resp = svc.files().list(
        q=f"(name contains '{q}' or fullText contains '{q}') and trashed=false",
        pageSize=max_results, orderBy="modifiedTime desc",
        fields="files(id,name,mimeType,modifiedTime,webViewLink)").execute()
    return [{"id": f["id"], "name": f["name"], "type": f["mimeType"],
             "modified": f.get("modifiedTime", ""), "link": f.get("webViewLink", "")}
            for f in resp.get("files", [])]


def drive_read(file_id: str, max_chars: int = 8000) -> str:
    svc = _get_service("drive", "v3")
    meta = svc.files().get(fileId=file_id, fields="name,mimeType").execute()
    mime = meta["mimeType"]
    if mime in _EXPORTABLE:
        data = svc.files().export(fileId=file_id, mimeType=_EXPORTABLE[mime]).execute()
    elif mime.startswith("application/vnd.google-apps"):
        return f"error: '{meta['name']}' is a {mime} — no text export available"
    else:
        data = svc.files().get_media(fileId=file_id).execute()
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
    return text[:max_chars] + ("\n…(truncated)" if len(text) > max_chars else "")


# -- Calendar ---------------------------------------------------------------

def calendar_list_events(days_ahead: int = 7) -> list[dict]:
    svc = _get_service("calendar", "v3")
    now = datetime.now().astimezone()
    resp = svc.events().list(
        calendarId="primary", timeMin=now.isoformat(),
        timeMax=(now + timedelta(days=days_ahead)).isoformat(),
        singleEvents=True, orderBy="startTime", maxResults=25).execute()
    return [{"id": e["id"], "title": e.get("summary", "(untitled)"),
             "start": e["start"].get("dateTime", e["start"].get("date")),
             "end": e["end"].get("dateTime", e["end"].get("date"))}
            for e in resp.get("items", [])]


def calendar_create_event(title: str, start_iso: str, end_iso: str, description: str = "") -> str:
    svc = _get_service("calendar", "v3")
    tz = datetime.now().astimezone().tzinfo
    body = {
        "summary": title, "description": description,
        "start": {"dateTime": datetime.fromisoformat(start_iso).replace(tzinfo=tz).isoformat()},
        "end": {"dateTime": datetime.fromisoformat(end_iso).replace(tzinfo=tz).isoformat()},
    }
    ev = svc.events().insert(calendarId="primary", body=body).execute()
    return f"created event: {ev.get('htmlLink', ev['id'])}"
