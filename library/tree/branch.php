<?php
/**
 * Branch is a node that may contain children nodes
 *
 * @author durso
 */

namespace library\tree;
use library\tree\node;
use library\utils;




class branch extends node{
    
     /**
     *
     * @var array list of children nodes  
     */
    protected $children = null;
    
    
    
    
     /*
     * 
     * Check if branch has child
     * @return boolean
     */
    public function hasChild(){
        return !is_null($this->children);
    }
    /*
     * 
     * Get all children nodes
     * @return array
     */
    public function getChildren(){
        return $this->children;
    }
    /*
     * 
     * Get a child
     * @return node
     */
    public function getChild($index){
        return $this->children[$index];
    }
    /*
     * 
     * Add a child to the branch
     * @param node $child the node object to be added as a child
     * @return void
     */
    public function addChild(node $child){
        $this->children[] = $child;
        $child->setParent($this);
    }
     /*
     * 
     * Remove a child from the branch
     * @param node $child child node to be removed
     * @return void
     */
    public function removeChild(node $child){
        $this->children = utils::array_remove($this->children,$child);
        $child->setParent();
    }
    

    
}
