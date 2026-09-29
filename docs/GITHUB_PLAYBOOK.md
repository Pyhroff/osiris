# GitHub Playbook — Concepts + What OSIRIS Actually Does

Two things in one document: git/GitHub concepts explained from scratch, and an exact reference
for what OSIRIS's tools do when you ask it to touch git or GitHub. Where a concept and a tool
meet, they're cross-referenced.

---

## 1. Quick answers first

**Is your git commit identity proper?** Yes, verified directly against the real commit log:
every commit is `Pyhroff <215876416+Pyhroff@users.noreply.github.com>`, and a full history
scan turns up zero instances of your real name anywhere. This matches your standing rule.

**Can OSIRIS make an existing private repo public?** Not yet — no tool for it exists today.
It's a genuinely one-way-feeling action worth building carefully rather than rushing: going
public exposes the *entire commit history* to everyone, permanently — flipping back to
private afterward does not undo the fact that it was public and could have been cloned/cached
in the meantime. If you want this built, say so explicitly — it would come with a hard
confirm-first requirement and a reminder to scrub any real name from history first (per your
standing rule), not silently flip visibility on request.

---

## 2. Git concepts, from the ground up

### Repository (repo)
A folder git is tracking. Contains your files (the **working directory**) plus a hidden `.git`
folder holding the entire history. `git_init` creates this — it's literally just running
`git init` and adding a safe default `.gitignore`.

### Commit
A saved snapshot of your files at a point in time, with a message describing why. Commits
chain together — each one points at its parent, forming your project's history. `git_commit`
does two things in sequence: **stage** everything (`git add -A`) then **commit** it
(`git commit -m message`).

### Staging area (the "index")
Between "I changed a file" and "it's part of history" there's a middle step: staging. You
choose what to include in the next commit. OSIRIS's `git_commit` always stages *everything*
changed (`add -A`) — there's no partial-stage tool, so if you only want some changes committed,
say so and it'll need to handle that more surgically (or you do it in your own terminal).

### HEAD
A pointer to "where you currently are" — normally the tip of whatever branch you're on. When
people say "detached HEAD," it means you've checked out a specific commit instead of a branch
tip; not something OSIRIS's tools put you into.

### Branch
A movable pointer to a commit — not a copy of your files, just a name that tracks "the latest
commit on this line of work." The default branch is usually `main` (older repos: `master`).
Branches exist so you can work on something without touching the stable line until it's ready.

**Why branch at all instead of always committing to main?** Two reasons: (1) if you're
experimenting and it doesn't work out, you delete the branch and `main` was never touched, and
(2) it's the *only* way pull requests make sense — a PR is fundamentally "here's what's
different between this branch and that one," so if you never left `main`, there's nothing to
compare.

`git_checkout(repo, branch, create=True)` is the tool: creates a new branch and switches to it
in one step (`git checkout -b branch-name`). Pass `create=False` to switch to a branch that
already exists instead of making a new one.

### Merge
Combining one branch's commits into another. On GitHub, this usually happens by **merging a
pull request** (see §4) rather than running `git merge` locally. Three merge strategies GitHub
offers when you click the merge button on a PR:
- **Merge commit** — keeps every individual commit from the branch, plus one new "merge"
  commit tying them together. Full history preserved, but can get noisy.
- **Squash and merge** — flattens the whole branch into one clean commit on `main`. Good
  default for feature branches with messy "wip" / "fix typo" commits along the way.
- **Rebase and merge** — replays the branch's commits on top of `main` individually, no merge
  commit at all. Keeps history linear but rewrites commit hashes.

For a personal project, **squash and merge** is usually the least to think about.

### Merge conflict
Happens when the same lines of a file were changed differently on both sides being merged. Git
can't guess which version you want, so it marks the file with `<<<<<<<` / `=======` / `>>>>>>>`
markers and stops, waiting for you to pick (or combine) the right content and commit the
result. OSIRIS has no conflict-resolution tool — if a merge conflicts, that's a "read the file,
decide, `fs_edit` it clean" job, same as any other file edit, with your explicit involvement.

### Remote
A repo living somewhere else that your local repo knows about — almost always `origin`,
pointing at GitHub. `git push` sends your local commits there; `git pull`/`fetch` brings
remote changes down. `git_sync_check` does a `fetch` then reports whether you're ahead,
behind, or diverged from `origin` — the tool to run before deciding whether to commit or pull.

### Clone vs. Fork
**Clone** = download a copy of a repo (yours or anyone's public one) to your machine, keeping
the connection to the original as `origin`. **Fork** = make your *own copy on GitHub* of
someone else's repo, so you can make changes and propose them back via a PR without having
write access to the original. OSIRIS doesn't fork — `github_create_repo` creates a fresh repo
from a local folder, it's not for contributing to someone else's project.

### `.gitignore`
A list of file patterns git should never track — `.env`, `node_modules/`, `__pycache__/`, etc.
`git_init` writes a sensible default one automatically (covers `.env`, credentials, common
build artifacts) specifically so a secret can't get committed by accident on a fresh repo.

---

## 3. GitHub-specific concepts

### Pull Request (PR)
A request to merge one branch into another, with a title, description, and a full diff anyone
can review, comment on, and approve before it lands. It is **not** the same as a commit or a
push — a PR is a *proposal*, reviewed on GitHub's website, that only actually changes `main`
once someone clicks **Merge**.

### What OSIRIS does with `github_create_pr` — precisely
```
github_create_pr(repo, title, body="", base="main")
```
Runs `gh pr create --title --body --base` from the local repo folder. That's it — **it creates
the PR, it does not merge it.** After this call:
- The PR exists on GitHub, open, unreviewed, unmerged
- OSIRIS gives you back the PR URL
- You (or a collaborator) review the diff on GitHub, then click **Merge pull request** —
  there's no `merge_pr` tool, that decision stays a manual step on GitHub's site
- Only `main` on GitHub is affected once merged — your local `main` needs a `git pull`
  afterward to catch up (also not automated; ask OSIRIS to `git_sync_check` and it'll tell you
  you're behind)

**The gap this closed today**: `github_create_pr` targets `base="main"` by default, but if you
never left `main` (no branch tool existed before), there'd be nothing to PR — GitHub refuses a
PR with zero commits between the two sides. `git_checkout` now exists specifically so this
works: branch first, commit on the branch, *then* PR.

**The actual workflow, phrase by phrase:**
```
"create a branch called add-auth in PromptStrike"      → git_checkout(create=True)
"make these changes ..."                                 → fs_edit, on that branch
"commit it"                                               → git_commit (on the branch)
"open a PR for this"                                       → github_create_pr(base="main")
```
If you just say "commit and push," without ever asking for a branch, it goes straight to
`main` — no PR, no review step, just a direct commit. That's still perfectly valid for solo
work; branches+PRs are for when you want a review checkpoint (even if the reviewer is just
future-you, a day later, with fresh eyes).

### Issues
A tracked to-do/bug/idea on a repo, separate from any code change. `github_list_issues` /
`github_create_issue` read and write these directly via the GitHub API — no branch or PR
involved, just a title + body landing on the repo's Issues tab.

### Visibility — public vs. private
Private = only you (and anyone you explicitly add) can see it. Public = anyone, indexed by
GitHub search, potentially cached/mirrored elsewhere even if you later make it private again.
`github_create_repo(repo, name, private=True)` defaults to private specifically so a build
never accidentally goes public. Changing an *existing* repo's visibility isn't built yet — see
§1.

### GitHub Actions / CI
Automated scripts that run on every push/PR (this very repo has one — `pytest` on
ubuntu+windows). OSIRIS doesn't create or read CI workflows; if you want a build/test pipeline
set up for a project, that's a manual `.github/workflows/*.yml` file you or I write directly,
not something the agent tools touch.

---

## 4. Every git/GitHub tool OSIRIS has, in one table

| Tool | What it actually runs | Confirm first? |
|---|---|---|
| `git_init` | `git init -b main` + writes a default `.gitignore` | no |
| `git_checkout` | `git checkout -b <branch>` (or plain checkout if `create=False`) | no |
| `git_status` | `git status --short --branch` | no |
| `git_diff` | `git diff --stat HEAD` (falls back to `--cached` on a brand-new repo) | no |
| `git_log` | `git log -N --oneline` | no |
| `git_commit` | `git add -A` then `git commit -m message` | **yes** |
| `git_push` | inline-token push (avoids the Windows credential-manager hang) | **yes** |
| `git_sync_check` | `fetch` + compares local vs. `origin`, reports ahead/behind/uncommitted | no |
| `git_list_repos` | scans your Desktop for every folder with a `.git`, reports branch + dirty count | no |
| `github_create_repo` | `gh repo create` (private by default) + inline-token push | **yes** |
| `github_create_pr` | `gh pr create --base main` — creates only, never merges | **yes** |
| `github_list_repos` | GitHub API: your 20 most-recently-pushed repos | no |
| `github_list_issues` | GitHub API: open issues on a repo | no |
| `github_create_issue` | GitHub API: opens a new issue | no |

"Confirm first" is enforced by the agent's persona instructions, not a hard code gate (except
`git_push`/`github_create_repo`'s underlying safety around credentials) — see HANDBOOK.md §12
for the honest caveat on what that does and doesn't guarantee.

---

## 5. Day-to-day phrasing

```
"check all my repos"                              → git_list_repos, one call, every repo
"what's changed in PromptStrike?"                 → git_diff, then explains it
"commit PromptStrike as 'fix scoring pipeline'"    → git_commit (confirms message first)
"push it"                                          → git_push (confirms first)
"create a branch called fix-auth"                  → git_checkout(create=True)
"open a PR for this"                               → github_create_pr (only makes sense post-branch)
"put this on github as private"                    → github_create_repo
"any open issues on Pyhroff/osiris?"               → github_list_issues
"file an issue: memory forget needs a dry-run"     → github_create_issue
"am I in sync with origin?"                        → git_sync_check
```
