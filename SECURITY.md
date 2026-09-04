# Security Policy

## Reporting a Vulnerability

If you find a security vulnerability in ACIES, please report it responsibly:

1. **Do NOT open a public issue**
2. Email: nicedev226@gmail.com
3. Include: description, steps to reproduce, potential impact

We will respond within 48 hours and work with you to fix the issue.

## Scope

ACIES is a research framework. It does not handle user data, authentication, or network communication. Security concerns are limited to:

- Input validation (malformed images, extreme values)
- Resource exhaustion (infinite loops, memory leaks)
- Supply chain (dependencies — ACIES uses stdlib only)

## Updates

Security fixes will be released as patch versions (e.g., v0.1.1).
