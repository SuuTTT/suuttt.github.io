---
title: "AutoSOTA Agent: Automated Blog & Paper Publishing with GitHub + Overleaf"
date: 2026-05-03
description: "AutoSOTA Agent: Automated Blog & Paper Publishing with GitHub + Overleaf"
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["autosota", "workflow", "automation", "github", "overleaf"]
---

katex: true
tags: ["autosota", "workflow", "automation", "github", "overleaf", "agent"]
---

{{< katex >}}

> AutoSOTA agents can now automatically publish papers to Overleaf and blog posts to your GitHub blog, with full support for KaTeX mathematical formulas and secure credential management via GCP Secret Manager.

---

## TL;DR

**AutoSOTA's new auto-push capability** lets agents automatically:

1. **Publish blog posts** to your GitHub blog (markdown with KaTeX formulas)
2. **Publish papers** to Overleaf (with LaTeX template support)
3. **Push code** from GPU instances without exposing private keys
4. **Securely manage credentials** via GCP Secret Manager

Example workflow:
```
Agent writes paper
    ↓
Auto-format with frontmatter
    ↓
Validate KaTeX formulas
    ↓
Push to Overleaf + GitHub
    ↓
Slack notification: "Paper published!"
```

**Key insight:** Use temporary GitHub tokens on rented instances (ephemeral hardware) and SSH keys only on your local machine (persistent, trusted). This keeps your credentials safe while enabling full automation.

---

## 1) The Problem: Manual Publishing Workflow

### 1.1 Traditional (Manual) Approach

When agents write papers and blog posts, you typically:

1. Agent writes paper.md on GPU instance
2. Copy output to local machine (manual)
3. Open Overleaf, create new project (manual)
4. Upload files to Overleaf (manual)
5. Create blog post, update frontmatter (manual)
6. Git commit and push (manual)
7. Post Slack notification (manual)

**Time cost:** 10-20 minutes per paper/post  
**Error rate:** High (forgot formulas? Missing frontmatter? Bad commit message?)  
**Security risk:** Copying files between machines, manual credential handling

### 1.2 The Automation Opportunity

The auto-push skill eliminates all manual steps:

$$
T_{\text{manual}} = T_{\text{write}} + T_{\text{format}} + T_{\text{upload}} + T_{\text{git}} + T_{\text{notify}}
$$

Becomes:

$$
T_{\text{auto}} = T_{\text{write}} + \Delta T_{\text{validation}}
$$

Where $\Delta T_{\text{validation}} \approx 2$ seconds (formula checks + format validation).

---

## 2) Architecture: Secure Multi-Environment Credential Management

### 2.1 The Challenge: Keys on Ephemeral Hardware

**Problem:** Vast.ai GPU instances are ephemeral. Private SSH keys shouldn't live there.

**Solution:** Use a 3-tier authentication strategy:

$$
\text{Auth Strategy} = \begin{cases}
\text{SSH key (local)} & \text{if } \text{machine} = \text{local, trusted} \\
\text{Temp token (30 min)} & \text{if } \text{machine} = \text{rented, ephemeral} \\
\text{GitHub App} & \text{if } \text{deployment} = \text{production}
\end{cases}
$$

### 2.2 Credential Storage: GCP Secret Manager

All credentials (GitHub, Overleaf, email) stored encrypted in GCP:

```
GCP Secret Manager
├── github-ssh-private-key      (SSH key, local only)
├── github-temp-token           (30-min expiry, rented instances)
├── overleaf-api-token          (Overleaf project access)
└── gmail-app-password          (Email notifications)
```

**Access pattern:**

```
Rented Instance: Load token from GCP → Use for 30 min → Token expires → Instance destroyed
Local Machine:   Load SSH key from GCP → Use indefinitely → Key stays encrypted
```

### 2.3 KaTeX Formula Validation Pipeline

Before publishing, the skill validates all formulas:

$$
\text{Valid Post} = \text{Frontmatter} \oplus \text{Content} \oplus \text{Formulas}_{\text{balanced}} \oplus \text{Links}_{\text{valid}}
$$

Where:
- $\oplus$ denotes conjunction (all must be true)
- $\text{Formulas}_{\text{balanced}}$: Check for unmatched `$$` delimiters
- $\text{Links}_{\text{valid}}$: Check for broken references

---

## 3) Implementation: The Auto-Push Skill

### 3.1 Blog Post Publishing

**Input:**
```python
publish_blog_post(
    title="My Research Findings",
    content="""
## Results

Our method achieves:
$$L = \\mathbb{E}[\\text{loss}]$$
""",
    tags=["research", "sota"],
    github_token="ghp_xxxx...",
    description="Research summary"
)
```

**Process:**

1. **Format frontmatter** (Hugo/Jekyll compatible):
```yaml
---
title: "My Research Findings"
date: 2026-05-03
tags: ["research", "sota"]
math: true
katex: true
---
```

2. **Validate formulas:**
   - Check for unmatched `$$` delimiters
   - Flag inline math issues
   - Ensure `{{< katex >}}` shortcode is present

3. **Create file path:**
```
content/projects/2026-05-03-my-research-findings.md
```

4. **Commit and push:**
```bash
git add content/projects/2026-05-03-my-research-findings.md
git commit -m "Blog: My Research Findings"
git push origin master
```

**Output:**
```python
{
    "status": "success",
    "url": "https://github.com/SuuTTT/suuttt.github.io",
    "commit": "a1b2c3d4",
    "message": "Blog: My Research Findings"
}
```

### 3.2 Paper Publishing to Overleaf

**Input:**
```python
publish_paper_to_overleaf(
    title="Fast RL with Structural Information",
    content=r"""
\documentclass{article}
\usepackage{amsmath}
\begin{document}

\section{Method}

We propose:
\begin{equation}
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^T \gamma^t r_t\right]
\end{equation}

\end{document}
""",
    overleaf_token="ol_xxxx..."
)
```

**Process:**

1. Create new Overleaf project via API
2. Upload LaTeX content as `main.tex`
3. Return shareable project link

### 3.3 Credential Management

**Three-tier auth system:**

```python
# Tier 1: Rented instance (temporary token)
setup_git_auth(
    auth_type="token",
    github_token=load_from_gcp_secret("github-temp-token")
)

# Tier 2: Local machine (persistent SSH key)
setup_git_auth(
    auth_type="ssh",
    ssh_key_path="~/.ssh/github_rsa"
)

# Tier 3: Production (GitHub App, auto-generated tokens)
setup_git_auth(
    auth_type="github-app",
    app_id=load_from_gcp_secret("github-app-id"),
    app_private_key=load_from_gcp_secret("github-app-key")
)
```

---

## 4) Workflow Integration

### 4.1 Automated Paper Publishing

**Workflow:** `write_paper_from_idea` + Auto-Publish

```yaml
stages:
  # ... write paper ...
  
  - id: publish_overleaf
    name: "Publish to Overleaf"
    skill: util-github-git-push
    inputs:
      title: ${stage_write.outputs.paper_title}
      content: ${stage_write.outputs.paper_latex}
      overleaf_token: ${secrets.OVERLEAF_API_TOKEN}
      paper_type: "arxiv"
    timeout: 2 minutes
  
  - id: publish_blog
    name: "Announce on Blog"
    skill: util-github-git-push
    inputs:
      title: ${stage_write.outputs.paper_title}
      content: "📄 New paper: [${stage_write.outputs.paper_title}](${stage_publish_overleaf.outputs.url})"
      tags: ["paper", "research"]
      github_token: ${secrets.GITHUB_TOKEN}
    timeout: 2 minutes
```

### 4.2 Automated Blog Publishing

**Workflow:** `write_blog_post` + Auto-Publish

```bash
sotaflow run write_paper_from_idea \
  --idea "GPU scheduling best practices" \
  --publish-blog true \
  --publish-overleaf true \
  --auth-type token
```

**Result:** 
- ✅ Paper on Overleaf
- ✅ Blog post on GitHub
- ✅ Slack notification with links

---

## 5) Security Best Practices

### 5.1 Credential Hierarchy

| Environment | Auth Method | Key Handling | Expiry |
|---|---|---|---|
| **Local machine** | SSH key | Long-lived, ~/.ssh/ | Never |
| **Rented GPU instance** | Temp token | 30-min, GCP Secret Mgr | Auto-revoke |
| **CI/CD pipeline** | GitHub App | Auto-generated per run | 1 hour |
| **Development** | Personal token | Short-lived, PAT settings | 30 days |

### 5.2 Token Rotation for Rented Instances

```bash
# Before running job on Vast.ai:
python scripts/create_temp_github_token.py \
  --expiry-minutes 60 \
  --scopes repo,workflow

# During job:
export GITHUB_TOKEN=$(gcloud secrets get-secret-version ...)
python github_git_push.py publish-blog ...

# After job (automatic):
python scripts/revoke_temp_tokens.py
# → Instance destroyed anyway, but cleanup is good practice
```

### 5.3 What NOT to Do

❌ **Don't:** Hardcode GitHub tokens in code  
❌ **Don't:** Store SSH keys on rented instances  
❌ **Don't:** Commit secrets to git (even in .gitignore)  
❌ **Don't:** Use long-lived PATs for ephemeral hardware  

✅ **Do:** Use GCP Secret Manager (encrypted, auditable)  
✅ **Do:** Use temporary tokens for rented hardware  
✅ **Do:** SSH keys on local machines only  
✅ **Do:** Rotate tokens regularly  

---

## 6) KaTeX Formula Rendering

### 6.1 Why Formula Rendering Matters

Blog posts with research often include math. The auto-push skill ensures formulas render correctly:

**Common issues fixed:**

1. **Unmatched delimiters:**
   ```markdown
   ❌ This formula: $x = y$$ renders incorrectly
   ✅ This formula: $x = y$ renders correctly
   ```

2. **Mixed delimiters:**
   ```markdown
   ❌ \( inline \) doesn't work in KaTeX
   ✅ $inline$ works correctly
   ```

3. **Block vs inline:**
   ```markdown
   ❌ Single $ on same line: $a + b = c$ and $x = y$ (hard to read)
   ✅ Block equations separate:
   $$a + b = c$$
   $$x = y$$
   ```

### 6.2 Validation Examples

**Example 1: Valid blog post**

```markdown
# My Post

The key result is:

$$
\\mathbb{E}[L] = \\sum_{i=1}^n \\text{loss}_i
$$

Where $L$ is the loss and $n$ is batch size.
```

**Validation:** ✅ All delimiters balanced, KaTeX shortcode present

**Example 2: Needs fixing**

```markdown
# My Post

The formula: $x + y$ seems off $z = w$ here.

$$L = \\text{loss}$

And another: \\(inline\\)
```

**Validation warnings:**
- ⚠️ Multiple inline formulas on same line (line 3)
- ⚠️ Unmatched `$$` (line 5)
- ⚠️ Found `\( \)` delimiters (use `$...$` instead, line 7)

---

## 7) Usage Examples

### 7.1 Local Machine: Publish Blog Post

```bash
cd /workspace/autosota-lite

# Load credentials
set -a && source .env.local && set +a
export GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-temp-token)

# Publish blog post
python plugins/autosota-lite/skills/util-key-manager/github_git_push.py publish-blog \
  --title "My Research Results" \
  --content "## Results\n\nOur method achieves 95% accuracy." \
  --tags research,sota \
  --github-token "$GITHUB_TOKEN" \
  --blog-repo "https://github.com/SuuTTT/suuttt.github.io.git" \
  --description "Research summary"
```

### 7.2 Rented GPU Instance: Auto-Publish Results

```bash
# Inside Vast.ai instance (after training)

# Load temporary credentials from GCP
export GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-temp-token)
export OVERLEAF_TOKEN=$(gcloud secrets versions access latest --secret=overleaf-api-token)

# Publish training results to blog
cat > publish_results.py << 'EOF'
from github_git_push import publish_blog_post
import json

with open("training_results.json") as f:
    results = json.load(f)

publish_blog_post(
    title=f"GPU Training Results - {results['date']}",
    content=f"""
## Training Results

- **Accuracy:** {results['accuracy']}%
- **Speed:** {results['steps_per_sec']} steps/sec
- **GPU Util:** {results['gpu_util']}%

$$L_{{final}} = {results['final_loss']:.6f}$$
""",
    tags=["gpu-training", "results"],
    github_token=os.getenv("GITHUB_TOKEN")
)
EOF

python publish_results.py
# ✅ Blog post published automatically
# Instance destroyed → token expires → no cleanup needed
```

### 7.3 Dry-run: Validate Before Publishing

```bash
# Test without actually pushing
python github_git_push.py publish-blog \
  --title "Test Post" \
  --content "Testing..." \
  --tags test \
  --github-token "$GITHUB_TOKEN" \
  --dry-run true

# Output: "Dry-run successful (would push abc1234 to master)"
```

---

## 8) Troubleshooting

### Issue: Formula rendering fails

**Symptom:** Formulas show as raw `$$..$$` on blog

**Solution:**
1. Check `math: true` in frontmatter
2. Check `{{< katex >}}` shortcode is present
3. Run validation: `python github_git_push.py validate --file post.md`
4. Fix unmatched delimiters

### Issue: Push fails with "Permission denied"

**Symptom:** `fatal: could not read from remote repository`

**Solution:**
- SSH: Check key permissions (`ssh-keyscan` vs SSH agent)
- Token: Check token expiry and scopes (`repo`, `workflow`)
- URL: Check URL format (should be `https://` or `git@github.com:user/repo.git`)

### Issue: GitHub API errors

**Symptom:** `Overleaf API error: 401 Unauthorized`

**Solution:**
- Verify token is not expired
- Check token scopes (need `docs`, `write` for Overleaf)
- Verify URL is correct
- Check rate limiting

---

## 9) Summary

**Auto-push skill enables:**
- ✅ Automatic blog publishing with KaTeX formula support
- ✅ Automatic paper publishing to Overleaf
- ✅ Secure credentials via GCP Secret Manager
- ✅ Safe usage on ephemeral hardware (temporary tokens)
- ✅ Formula validation before publishing
- ✅ Integration with AutoSOTA workflows

**Key insight:** Credentials are tier-aware. Local machines use persistent keys. Rented instances use temporary tokens. Both stored safely in GCP Secret Manager, never exposed in code.

**Next:** Use this in your `write_paper_from_idea` and `write_blog_post` workflows for end-to-end automation!

---

## References

- [GitHub Personal Access Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
- [Overleaf API Documentation](https://www.overleaf.com/learn/how-to/Overleaf_API)
- [KaTeX Documentation](https://katex.org/docs/supported.html)
- [GCP Secret Manager](https://cloud.google.com/secret-manager/docs)
- [Your blog repository](https://github.com/SuuTTT/suuttt.github.io)
