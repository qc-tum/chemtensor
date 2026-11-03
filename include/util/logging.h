/// \file logging.h
/// \brief Logging and progress reporting.

#include <stdio.h>


//________________________________________________________________________________________________________________________
///
/// \brief Verbosity level.
///
enum verbosity_level
{
	CT_VERBOSITY_NONE   = 0,  //!< remain silent
	CT_VERBOSITY_BASIC  = 1,  //!< report intermediate steps and basic information in a calculation
	CT_VERBOSITY_TIMING = 2,  //!< additionally include timing information
};


extern enum verbosity_level ct_verbosity;


//________________________________________________________________________________________________________________________
//


extern FILE* fd_log;


void logprint(char const *fmt, ...);

void logerror(char const *fmt, ...);
