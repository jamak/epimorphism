'''OpenGL extension ARB.vertex_buffer_object

This module customises the behaviour of the
OpenGL.raw.GL.ARB.vertex_buffer_object to provide a more
Python-friendly API
'''
from OpenGL import platform, constants, constant, arrays
from OpenGL import extensions, wrapper
from OpenGL.GL import glget
import ctypes
from OpenGL.raw.GL.ARB.vertex_buffer_object import *
### END AUTOGENERATED SECTION

#glDeleteBuffersARB = arrays.setInputArraySizeType(
#       glDeleteBuffersARB,
#       None,
#       arrays.GLuintArray,
#       'buffers',
#)

#glGenBuffersARB = wrapper.wrapper( glGenBuffersARB ).setOutput(
#       'buffers', lambda n: (n,), 'n',
#)

#def _sizeOfArrayInput( pyArgs, index, wrapper ):
#       return (
#               arrays.ArrayDatatype.arrayByteCount( pyArgs[index] )
#       )

#glBufferDataARB = wrapper.wrapper( glBufferDataARB ).setPyConverter(
#       'data', arrays.asVoidArray(),
#).setPyConverter( 'size' ).setCResolver(
#       'data', arrays.ArrayDatatype.voidDataPointer ,
#).setCConverter(
#       'size', _sizeOfArrayInput,
#).setReturnValues(
#       wrapper.returnPyArgument( 'data' )
#)

#glBufferSubDataARB = wrapper.wrapper( glBufferSubDataARB ).setPyConverter(
#       'data', arrays.asVoidArray(),
#).setPyConverter( 'size' ).setCResolver(
#       'data', arrays.ArrayDatatype.voidDataPointer ,
#).setCConverter(
#       'size', _sizeOfArrayInput,
#).setReturnValues(
#       wrapper.returnPyArgument( 'data' )
#)
