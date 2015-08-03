<?php
/**
 * Description of node
 *
 * @author durso
 */

namespace library\tree;
use library\tree\node;
use library\utils;

class branch extends node{
    
     /**
     *
     * @var tree list of children nodes  
     */
    protected $children = null;
    
    
    
    
     /*
     * 
     * Check if node has children
     * @return boolean
     */
    public function hasChild(){
        return !is_null($this->children);
    }
    /*
     * 
     * Get all children
     * @return node
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
     * Add a child to the node
     * @param tree $child the object to be added as a child
     * @return void
     */
    public function addChild(node $child){
        $this->children[] = $child;
        $child->setParent($this);
    }
     /*
     * 
     * Remove a child from the node
     * @param tree $child child node to be removed
     * @return void
     */
    public function removeChild(node $child){
        $this->children = utils::array_remove($this->children,$child);
        $child->setParent(null);
    }
    
    public function getChildrenElements(){
        assert(!empty($this->children));
        $list = array();
        foreach($this->children as $child){
            $element = $child->getValue();
            if($element->hasTag()){
                $list[] = $child;
            }
        }
        return $list;
    }
    
    public function getChildrenByValue($method,$args){
        assert(!empty($this->children));
        $list = array();
        foreach($this->children as $child){
            $element = $child->getValue();
            if($element->hasTag()){
                if($element->$method($args)){
                    $list[] = $child;
                }
            }
        }
        return $list;
    }
    
    
}
