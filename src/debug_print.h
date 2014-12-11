#ifndef __DEBUG_PRINT__
#define __DEBUG_PRINT__

#if 1
#include <iostream>
#define DE(X) std::cout<<#X<<":"<<X<<std::endl
#define D(X) std::cout<<":"<<X<<" "
#define DX(X) std::cout<<#X<<":"<<X<<" "
#define EE std::cout<<"\n"
#else
#define DE(X)
#define D(X) 
#define DX(X)
#define EE
#endif

#endif //__DEBUG_PRINT__
