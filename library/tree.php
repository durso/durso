<?php
/**
 * Tree list
 *
 * @author durso
 */
namespace library;
use library\utils;

abstract class tree {
    /**
     *
     * @var tree list of children nodes  
     */
    protected $children = null;
    /**
     *
     * @var tree parent node  
     */
    protected $parent = null;
     /**
     *
     * @var string node value
     */
    protected $value = "";
     /**
     *
     * @var tree ancestors
     */
    protected $ancestors = array();
    /*
     * 
     * Check if node has children
     * @return boolean
     */
    public function hasChildren(){
        return !is_null($this->children);
    }
    /*
     * 
     * Get all children
     * @return boolean
     */
    public function getChildren(){
        return $this->children;
    }
    /*
     * 
     * Check if node has a parent
     * @return boolean
     */
    public function hasParent(){
        return !is_null($this->parent);
    }
     /*
     * 
     * Set the node parent
     * @param tree $parent
     * @return void
     */
    public function setParent(tree $parent){
        $this->parent = $parent;
    }
     /*
     * 
     * Get the node parent
     * @return tree 
     */
    public function getParent(){
       return $this->parent;
    }
    /*
     * 
     * Add a child to the node
     * @param tree $child the object to be added as a child
     * @return void
     */
    public function addChild(tree $child){
        $this->children[] = $child;
        $child->setParent($this);
    }
     /*
     * 
     * Remove a child from the node
     * @param tree $child child node to be removed
     * @return void
     */
    public function removeChild(tree $child){
        $this->children = utils::array_remove($this->children,$child);
        $child->setParent(null);
    }
    public function setValue($value){
        $this->value = $value;
    }
    public function getValue(){
        return $this->value;
    }
    public function getAncestors(){
        $ancestors = array();
        $node = $this;
        while (true) {
            $parent = $node->getParent();
            $node = buildList($parent,$ancestors);
            if(!$node){
                break;
            }
        }
        $this->ancestors = $ancestors;
        return $this->ancestors;
    }
    public function getAncestorsList(){
        return $this->ancestors; 
    }
    public function searchAncestorsProperty($method){
        $ancestors = array();
        $node = $this;
        while (true) {
            $parent = $node->getParent();
            $node = $this->buildList($parent,$ancestors);
            if(!$node || $parent->$method()){
                break;
            }
        }
        return $ancestors;
    }
    protected function buildList(tree $parent, &$ancestors){
        if($parent){
            array_unshift($ancestors, $parent);
            $node = $parent;
            return $node;
        }
        return false;
    }
    protected function getSiblings(){
       $parent = $this->getParent();
       if($parent){
           return $parent->getChildren();
       }
       return false;
    }
    protected function hasSiblings(){
       return count($this->getSiblings()) > 1;
    }
    protected function getSiblingsIndex(){
        $siblings = $this->getSiblings();
        if(!$siblings){
            return -1;
        }
        $i = 0;
        foreach($siblings as $element){    
            if($element == $this){
                return $i;
            }
            $i++;
        }
    }
}
