#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>  
#include <WinSock2.h> 
#include <WS2tcpip.h> 
#include <iphlpapi.h> 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/applink.c>
#include <openssl/x509v3.h>

#pragma comment(lib,"ws2_32.lib")
#pragma comment(lib,"libcrypto.lib")
#pragma comment(lib,"libssl.lib")

static BOOL WINAPI CtrlHandler(DWORD CtrlType);
HANDLE hEvent = INVALID_HANDLE_VALUE;
BIO* gpBio = NULL;

[[noreturn]] void print_errors_and_exit(const char* message)
{
	fprintf(stderr, "%s\n", message);
	ERR_print_errors_fp(stderr);
	getchar();
	exit(1);
}

std::string receive_some_data(BIO* bio)
{
	char buffer[1024] = { 0 };

	int len = BIO_read(bio, buffer, sizeof(buffer));
	if (len < 0)
	{
		printf("CLIENT:: error in BIO_read\n");
	}
	else if (len > 0)
	{
		return std::string(buffer, len);
	}
	else if (BIO_should_retry(bio))
	{
		return receive_some_data(bio);
	}
	else
	{
		printf("CLIENT:: empty BIO_read");
	}

	return nullptr;
}

std::vector<std::string> split_headers(const std::string& text)
{
	std::vector<std::string> lines;
	const char* start = text.c_str();
	while (const char* end = strstr(start, "\r\n"))
	{
		lines.push_back(std::string(start, end));
		start = end + 2;
	}
	return lines;
}

std::string receive_http_message(BIO* bio)
{
	size_t content_length = 0;

	std::string headers = receive_some_data(bio);
	if (0 == headers.size())
	{
		printf("CLIENT:: receive_http_message() failed.\n");
	}

	char* end_of_headers = strstr(&headers[0], "\r\n\r\n");

	while (end_of_headers == nullptr)
	{
		headers += receive_some_data(bio);
		end_of_headers = strstr(&headers[0], "\r\n\r\n");
	}

	std::string body = std::string(end_of_headers + 4, &headers[headers.size()]);

	headers.resize(end_of_headers + 2 - &headers[0]);

	for (const std::string& line : split_headers(headers))
	{
		if (const char* colon = strchr(line.c_str(), ':'))
		{
			auto header_name = std::string(&line[0], colon);
			if (header_name == "Content-Length")
			{
				content_length = std::stoul(colon + 1);
			}
		}
	}

	while (body.size() < content_length)
	{
		body += receive_some_data(bio);
	}

	return headers + "\r\n" + body;
}

void send_http_request(BIO* bio, const std::string& line, const std::string& host)
{
	std::string request = line + "\r\n";
	request += "Host: " + host + "\r\n";
	request += "\r\n";

	BIO_write(bio, request.data(), (int)request.size());
	BIO_flush(bio);
}

SSL* get_ssl(BIO* bio)
{
	SSL* ssl = nullptr;
	BIO_get_ssl(bio, &ssl);
	if (ssl == nullptr)
	{
		print_errors_and_exit("HTTP CLIENT:: Error in BIO_get_ssl");
	}
	return ssl;
}

void verify_the_certificate(SSL* ssl, const std::string& expected_hostname)
{
	X509* cert = SSL_get_peer_certificate(ssl);
	if (cert == nullptr) {
		fprintf(stderr, "No certificate was presented by the server\n");
		exit(1);
	}
#if OPENSSL_VERSION_NUMBER < 0x10100000L
	if (X509_check_host(cert, expected_hostname.data(), expected_hostname.size(), 0, nullptr) != 1) {
		fprintf(stderr, "Certificate verification error: X509_check_host\n");
		exit(1);
	}
#else
	// X509_check_host is called automatically during verification,
	// because we set it up in main().
	(void)expected_hostname;
#endif

	X509_print_fp(stdout, cert);

	int err = SSL_get_verify_result(ssl);
	if (err != X509_V_OK) {
		const char* message = X509_verify_cert_error_string(err);
		fprintf(stderr, "Certificate verification error: %s (%d)\n", message, err);
		exit(1);
	}
}

int main()
{
	int iRet;

	/* Set up the SSL context */
	SSL_CTX* pSSL_Ctx = SSL_CTX_new(TLS_client_method());


	if (SSL_CTX_set_default_verify_paths(pSSL_Ctx) != 1)
	{
		print_errors_and_exit("HTTP::CLIENT Error setting up trust store");
	}

	BIO* pBio = BIO_new_connect("en.wikipedia.org:443");
	if (pBio == nullptr)
	{
		print_errors_and_exit("HTTPS::CLIENT Error in BIO_new_connect");
	}

	if (BIO_do_connect(pBio) <= 0)
	{
		print_errors_and_exit("HTTPS::CLIENT Error in BIO_do_connect");
	}

	BIO* pSSL_NEW_Bio = BIO_new_ssl(pSSL_Ctx, 1);
	if (pSSL_NEW_Bio == nullptr)
	{
		print_errors_and_exit("HTTPS::CLIENT Error in BIO_new_ssl");
	}

	BIO* pSSL_Bio = BIO_push(pSSL_NEW_Bio, pBio);
	if (pSSL_Bio == nullptr)
	{
		print_errors_and_exit("HTTPS::CLIENT Error in BIO_push");
	}

	iRet = SSL_set_tlsext_host_name(get_ssl(pSSL_Bio), "en.wikipedia.org");
	if (iRet == 0)
	{
		printf("SSL_set_tlsext_host_name() FAILED \n");
	}

	if (BIO_do_handshake(pSSL_Bio) <= 0)
	{
		print_errors_and_exit("HTTPS::CLIENT Error in BIO_do_handshake");
	}

	//verify_the_certificate(get_ssl(pSSL_Bio), "localhost:8080");

	send_http_request(pSSL_Bio, "GET /wiki/Sun HTTP/1.1", "en.wikipedia.org");

	std::string response = receive_http_message(pSSL_Bio);

	printf("%s", response.c_str());

	return 0;
}