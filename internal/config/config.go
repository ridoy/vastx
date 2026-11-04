// Package config loads storage credentials from a JSON file and environment
// variables. Environment variables take precedence over the file.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// EnvConfigPath overrides the config file location.
const EnvConfigPath = "VASTX_CONFIG"

// Config holds the settings needed to talk to the object store.
type Config struct {
	AccountID       string `json:"account_id,omitempty"`
	AccessKeyID     string `json:"access_key_id,omitempty"`
	SecretAccessKey string `json:"secret_access_key,omitempty"`
	Bucket          string `json:"bucket,omitempty"`
	// Endpoint overrides the R2 endpoint derived from AccountID. Any
	// S3-compatible endpoint works.
	Endpoint string `json:"endpoint,omitempty"`
	Region   string `json:"region,omitempty"`
}

type field struct {
	key    string
	env    string
	secret bool
	ptr    func(*Config) *string
}

var fields = []field{
	{"account_id", "VASTX_R2_ACCOUNT_ID", false, func(c *Config) *string { return &c.AccountID }},
	{"access_key_id", "VASTX_R2_ACCESS_KEY_ID", false, func(c *Config) *string { return &c.AccessKeyID }},
	{"secret_access_key", "VASTX_R2_SECRET_ACCESS_KEY", true, func(c *Config) *string { return &c.SecretAccessKey }},
	{"bucket", "VASTX_R2_BUCKET", false, func(c *Config) *string { return &c.Bucket }},
	{"endpoint", "VASTX_S3_ENDPOINT", false, func(c *Config) *string { return &c.Endpoint }},
	{"region", "VASTX_S3_REGION", false, func(c *Config) *string { return &c.Region }},
}

// Keys returns the settable config keys.
func Keys() []string {
	keys := make([]string, len(fields))
	for i, f := range fields {
		keys[i] = f.key
	}
	return keys
}

// Path returns the config file location.
func Path() (string, error) {
	if p := os.Getenv(EnvConfigPath); p != "" {
		return p, nil
	}
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("locate config dir: %w", err)
		}
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, "vastx", "config.json"), nil
}

// LoadFile reads the config file. A missing file yields an empty config.
func LoadFile(path string) (Config, error) {
	var c Config
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(b, &c); err != nil {
		return c, fmt.Errorf("parse %s: %w", path, err)
	}
	return c, nil
}

// Load reads the config file and applies environment overrides.
func Load() (Config, error) {
	p, err := Path()
	if err != nil {
		return Config{}, err
	}
	c, err := LoadFile(p)
	if err != nil {
		return c, err
	}
	for _, f := range fields {
		if v := os.Getenv(f.env); v != "" {
			*f.ptr(&c) = v
		}
	}
	return c, nil
}

// Save writes the config file with owner-only permissions.
func Save(path string, c Config) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Set assigns a value by key name.
func (c *Config) Set(key, value string) error {
	for _, f := range fields {
		if f.key == key {
			*f.ptr(c) = strings.TrimSpace(value)
			return nil
		}
	}
	return fmt.Errorf("unknown config key %q (valid: %s)", key, strings.Join(Keys(), ", "))
}

// Entry is one row of a config listing.
type Entry struct {
	Key, Value, Env string
}

// Entries lists all settings with secrets redacted.
func (c Config) Entries() []Entry {
	out := make([]Entry, 0, len(fields))
	for _, f := range fields {
		v := *f.ptr(&c)
		if f.secret && v != "" {
			v = redact(v)
		}
		out = append(out, Entry{Key: f.key, Value: v, Env: f.env})
	}
	return out
}

func redact(s string) string {
	if len(s) <= 8 {
		return "********"
	}
	return s[:4] + strings.Repeat("*", 8) + s[len(s)-4:]
}

// EndpointURL returns the S3 endpoint.
func (c Config) EndpointURL() string {
	if c.Endpoint != "" {
		return c.Endpoint
	}
	return "https://" + c.AccountID + ".r2.cloudflarestorage.com"
}

// RegionOrDefault returns the configured region, or "auto" as R2 expects.
func (c Config) RegionOrDefault() string {
	if c.Region != "" {
		return c.Region
	}
	return "auto"
}

// Validate reports missing settings.
func (c Config) Validate() error {
	var missing []string
	for _, f := range fields {
		switch f.key {
		case "endpoint", "region":
			continue
		case "account_id":
			if c.Endpoint != "" {
				continue
			}
		}
		if *f.ptr(&c) == "" {
			missing = append(missing, fmt.Sprintf("%s ($%s)", f.key, f.env))
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing storage config: %s\nset them with `vastx config set <key> <value>` or the environment variables",
			strings.Join(missing, ", "))
	}
	return nil
}
