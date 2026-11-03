/// \file logging.c
/// \brief Logging and progress reporting.

#include "logging.h"
#include <stdarg.h>


/// \brief Global verbosity level.
enum verbosity_level ct_verbosity = 0;


/// \brief File descriptor of log file.
FILE* fd_log = NULL;


//________________________________________________________________________________________________________________________
///
/// \brief Print both to standard output and a log file, using the syntax as 'printf(...)'.
///
void logprint(char const *fmt, ...)
{
	va_list ap;

	// standard printf()
	va_start(ap, fmt);
	vprintf(fmt, ap);
	fflush(stdout);
	va_end(ap);

	// write same output to the file specified by 'fd_log'
	if (fd_log != NULL)
	{
		va_start(ap, fmt);
		vfprintf(fd_log, fmt, ap);
		fflush(fd_log);
		va_end(ap);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Print both to standard error output and a log file, using the syntax as 'fprintf(stderr, ...)'.
///
void logerror(char const *fmt, ...)
{
	va_list ap;

	// standard printf()
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	// write same output to the file specified by 'fd_log'
	if (fd_log != NULL)
	{
		va_start(ap, fmt);
		vfprintf(fd_log, fmt, ap);
		fflush(fd_log);
		va_end(ap);
	}
}
