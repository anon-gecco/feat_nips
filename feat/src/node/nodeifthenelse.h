/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_IFTHENELSE
#define NODE_IFTHENELSE

#include "node.h"

namespace FT{
	class NodeIfThenElse : public Node
    {
    	public:
    	
    		NodeIfThenElse()
    	    {
    			name = "ite";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 1;
    			complexity = 5;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack)
            {
                ArrayXd f1 = stack.f.pop();
                ArrayXd f2 = stack.f.pop();
                stack.f.push(limited(stack.b.pop().select(f1,f2)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("if-then-else(" + stack.bs.pop() + "," + stack.fs.pop() + "," + stack.fs.pop() + ")");
            }
        protected:
            NodeIfThenElse* clone_impl() const override { return new NodeIfThenElse(*this); };  
            NodeIfThenElse* rnd_clone_impl() const override { return new NodeIfThenElse(); };  
    };

}	

#endif
