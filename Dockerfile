FROM python:3.12-slim AS python-base
WORKDIR /app
COPY acies/ ./acies/
COPY examples/ ./examples/
COPY test_apc.py .
RUN python3 test_apc.py

FROM golang:1.22-bookworm AS go-build
WORKDIR /src
COPY go.mod .
COPY core.go main.go .
RUN go build -o /acies-cli .

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=python-base /usr/local/lib/python3.12/ /usr/local/lib/python3.12/
COPY --from=python-base /usr/local/bin/python3 /usr/local/bin/python3
COPY --from=go-build /acies-cli /usr/local/bin/acies-cli
COPY acies/ ./acies/
COPY examples/ ./examples/
COPY cpp/libacies.so /usr/local/lib/
COPY test_apc.py .
RUN ldconfig

ENTRYPOINT ["acies-cli"]
CMD ["--help"]
