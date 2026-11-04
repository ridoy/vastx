// Package store wraps an S3-compatible bucket (Cloudflare R2 by default) and
// defines how models are laid out in it:
//
//	<name>/<file path>                 model files
//	<name>/.vastx/manifest.json        model manifest
//	.vastx/shares/<id>.json            share manifests with presigned URLs
//	.vastx/bin/<hash>/vastx-linux-*    vastx binaries for remote bootstrap
package store

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/ratelimit"
	"github.com/aws/aws-sdk-go-v2/aws/retry"
	awshttp "github.com/aws/aws-sdk-go-v2/aws/transport/http"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"

	"github.com/ridoy/vastx/internal/config"
	"github.com/ridoy/vastx/internal/manifest"
)

// ErrNotFound is returned when an object or model doesn't exist.
var ErrNotFound = errors.New("not found")

// MaxPresignTTL is the SigV4 limit on presigned URL lifetime.
const MaxPresignTTL = 7 * 24 * time.Hour

// ManifestKey returns the key of a model's manifest.
func ManifestKey(name string) string { return name + "/" + manifest.MetaDir + "/manifest.json" }

// ObjectKey returns the key of a file inside a model.
func ObjectKey(name, rel string) string { return name + "/" + rel }

// Store is a handle on the bucket.
type Store struct {
	client  *s3.Client
	presign *s3.PresignClient
	bucket  string
}

// New creates a Store. maxConns sizes the HTTP connection pool.
func New(cfg config.Config, maxConns int) (*Store, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	maxConns = max(maxConns, 16)
	hc := awshttp.NewBuildableClient().WithTransportOptions(func(t *http.Transport) {
		t.MaxIdleConns = maxConns * 2
		t.MaxIdleConnsPerHost = maxConns * 2
		// One TCP connection per in-flight request. HTTP/2 would multiplex
		// all parts over a single connection and cap throughput.
		t.ForceAttemptHTTP2 = false
		t.TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{}
		t.WriteBufferSize = 256 << 10
		t.ReadBufferSize = 256 << 10
	})
	client := s3.New(s3.Options{
		Region:       cfg.RegionOrDefault(),
		BaseEndpoint: aws.String(cfg.EndpointURL()),
		Credentials: aws.NewCredentialsCache(
			credentials.NewStaticCredentialsProvider(cfg.AccessKeyID, cfg.SecretAccessKey, "")),
		UsePathStyle: true,
		HTTPClient:   hc,
		// vastx keeps its own SHA-256 per chunk. The SDK's default CRC
		// checksums aren't supported uniformly across S3-compatible stores.
		RequestChecksumCalculation: aws.RequestChecksumCalculationWhenRequired,
		ResponseChecksumValidation: aws.ResponseChecksumValidationWhenRequired,
		Retryer: retry.NewStandard(func(o *retry.StandardOptions) {
			o.MaxAttempts = 10
			o.MaxBackoff = 30 * time.Second
			o.RateLimiter = ratelimit.None
		}),
	})
	return &Store{client: client, presign: s3.NewPresignClient(client), bucket: cfg.Bucket}, nil
}

// Bucket returns the bucket name.
func (s *Store) Bucket() string { return s.bucket }

// IsNotFound reports whether err means the object doesn't exist.
func IsNotFound(err error) bool {
	if errors.Is(err, ErrNotFound) {
		return true
	}
	var nsk *types.NoSuchKey
	var nf *types.NotFound
	if errors.As(err, &nsk) || errors.As(err, &nf) {
		return true
	}
	var ae smithy.APIError
	if errors.As(err, &ae) {
		switch ae.ErrorCode() {
		case "NoSuchKey", "NotFound", "404":
			return true
		}
	}
	var re *awshttp.ResponseError
	return errors.As(err, &re) && re.HTTPStatusCode() == http.StatusNotFound
}

// GetBytes downloads a whole object.
func (s *Store) GetBytes(ctx context.Context, key string) ([]byte, error) {
	out, err := s.client.GetObject(ctx, &s3.GetObjectInput{Bucket: &s.bucket, Key: &key})
	if err != nil {
		if IsNotFound(err) {
			return nil, fmt.Errorf("%s: %w", key, ErrNotFound)
		}
		return nil, fmt.Errorf("get %s: %w", key, err)
	}
	defer out.Body.Close()
	return io.ReadAll(out.Body)
}

// PutBytes uploads a whole object in one request.
func (s *Store) PutBytes(ctx context.Context, key string, b []byte, contentType string) error {
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:        &s.bucket,
		Key:           &key,
		Body:          bytes.NewReader(b),
		ContentLength: aws.Int64(int64(len(b))),
		ContentType:   &contentType,
	})
	if err != nil {
		return fmt.Errorf("put %s: %w", key, err)
	}
	return nil
}

// Head returns an object's size, or exists=false.
func (s *Store) Head(ctx context.Context, key string) (size int64, exists bool, err error) {
	out, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{Bucket: &s.bucket, Key: &key})
	if err != nil {
		if IsNotFound(err) {
			return 0, false, nil
		}
		return 0, false, fmt.Errorf("head %s: %w", key, err)
	}
	return aws.ToInt64(out.ContentLength), true, nil
}

// Delete removes an object. Deleting a missing object is not an error.
func (s *Store) Delete(ctx context.Context, key string) error {
	_, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{Bucket: &s.bucket, Key: &key})
	if err != nil && !IsNotFound(err) {
		return fmt.Errorf("delete %s: %w", key, err)
	}
	return nil
}

// List calls fn for each object under prefix.
func (s *Store) List(ctx context.Context, prefix string, fn func(key string, size int64) error) error {
	p := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{Bucket: &s.bucket, Prefix: &prefix})
	for p.HasMorePages() {
		page, err := p.NextPage(ctx)
		if err != nil {
			return fmt.Errorf("list %s: %w", prefix, err)
		}
		for _, o := range page.Contents {
			if err := fn(aws.ToString(o.Key), aws.ToInt64(o.Size)); err != nil {
				return err
			}
		}
	}
	return nil
}

// CreateMultipart starts a multipart upload and returns its ID.
func (s *Store) CreateMultipart(ctx context.Context, key string) (string, error) {
	out, err := s.client.CreateMultipartUpload(ctx, &s3.CreateMultipartUploadInput{
		Bucket:      &s.bucket,
		Key:         &key,
		ContentType: aws.String("application/octet-stream"),
	})
	if err != nil {
		return "", fmt.Errorf("start upload of %s: %w", key, err)
	}
	return aws.ToString(out.UploadId), nil
}

// UploadPart uploads part number part (1-based) and returns its ETag.
func (s *Store) UploadPart(ctx context.Context, key, uploadID string, part int32, b []byte) (string, error) {
	out, err := s.client.UploadPart(ctx, &s3.UploadPartInput{
		Bucket:        &s.bucket,
		Key:           &key,
		UploadId:      &uploadID,
		PartNumber:    aws.Int32(part),
		Body:          bytes.NewReader(b),
		ContentLength: aws.Int64(int64(len(b))),
	})
	if err != nil {
		return "", fmt.Errorf("upload %s part %d: %w", key, part, err)
	}
	return aws.ToString(out.ETag), nil
}

// CompleteMultipart finishes a multipart upload. etags are in part order.
func (s *Store) CompleteMultipart(ctx context.Context, key, uploadID string, etags []string) error {
	parts := make([]types.CompletedPart, len(etags))
	for i, e := range etags {
		parts[i] = types.CompletedPart{ETag: aws.String(e), PartNumber: aws.Int32(int32(i + 1))}
	}
	_, err := s.client.CompleteMultipartUpload(ctx, &s3.CompleteMultipartUploadInput{
		Bucket:          &s.bucket,
		Key:             &key,
		UploadId:        &uploadID,
		MultipartUpload: &types.CompletedMultipartUpload{Parts: parts},
	})
	if err != nil {
		return fmt.Errorf("complete upload of %s: %w", key, err)
	}
	return nil
}

// AbortMultipart cancels a multipart upload and frees its parts.
func (s *Store) AbortMultipart(ctx context.Context, key, uploadID string) error {
	_, err := s.client.AbortMultipartUpload(ctx, &s3.AbortMultipartUploadInput{
		Bucket: &s.bucket, Key: &key, UploadId: &uploadID,
	})
	return err
}

// PresignGet returns a URL that allows GET (including ranged GET) on key
// without credentials until ttl elapses.
func (s *Store) PresignGet(ctx context.Context, key string, ttl time.Duration) (string, error) {
	req, err := s.presign.PresignGetObject(ctx, &s3.GetObjectInput{Bucket: &s.bucket, Key: &key},
		s3.WithPresignExpires(ttl))
	if err != nil {
		return "", fmt.Errorf("presign %s: %w", key, err)
	}
	return req.URL, nil
}
