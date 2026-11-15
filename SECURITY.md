# Security and credential handling

Never commit API keys, access tokens, provider responses, private result archives, or
local configuration files. Copy config.example.yaml to config.yaml, fill in the
credentials locally, and keep the ignored config.yaml out of Git.

If a credential is accidentally exposed, revoke it at the provider immediately and
replace it with a new key. Please report suspected leaks privately rather than
opening an issue containing the secret.

