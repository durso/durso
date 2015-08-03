<?php
/**
 * Tree list
 *
 * @author durso
 */
namespace library\tree;
use library\dom\object;


abstract class node {
   
    /**
     *
     * @var parent node  
     */
    protected $parent = null;

    protected $value;
    
    public function __construct(object $object){
	$this->value = $object;
    }
    public function getValue(){
	return $this->value;
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
     * @param object $parent
     * @return void
     */
    public function setParent(node $node){
	$this->parent = $node;
    }
     /*
     * 
     * Get the node parent
     * @return tree 
     */
    public function getParent(){
       return $this->parent;
    }
    
    public function getAncestor($index){
        $node = $this;
        $i = 1;
        while (true) {
            $node = $node->getParent();
            if(!$node){
                break;
            }
            if($i == $index){
                return $node;
            }
            $i++;
        }
        return false;
    }

    public function getAncestors(){
        $ancestors = array();
        $node = $this;
        while (true) {
            $parent = $node->getParent();
            if(!$parent){
                break;
            }
            $ancestors[] = $parent;
            $node = $parent;
        }
        return $ancestors;

    }

    
    public function getAncestorByValue($method,$arg){
        $node = $this;
        while($node->hasParent()){
            $parent = $node->getParent();
            if($parent->getValue()->$method($arg)){
                return $parent;
            }
            $node = $parent;
        }
        return false;
    }
    
    public function getAncestorsByValue($method,$arg){
        $list = array();
        $node = $this;
        while($node->hasParent()){
            $parent = $node->getParent();
            if($parent->getValue()->$method($arg)){
                $list[] = $parent;
            }
            $node = $parent;
        }
        return $list;
    }
    
    public function searchAncestorsProperty($method){
        $ancestors = array();
        $node = $this;
        while ($node->hasParent()) {
            $parent = $node->getParent();
            $node = $this->buildList($parent,$ancestors);
            if(!$node || $parent->getValue()->$method()){
                break;
            }
        }
        return $ancestors;
    }
    protected function buildList($parent, &$ancestors){
        if($parent){
            array_unshift($ancestors, $parent);
            $node = $parent;
            return $node;
        }
        return false;
    }
    
    //siblings is returning the element itself
    public function getSiblings($tagOnly = true){
       $self = $this;
       $parent = $this->getParent();
       if($parent){
           if(!$tagOnly){
                $children = $parent->getChildren();
                return utils::array_remove($siblings,$this);
           }
           $children = $parent->getChildren();
           $array = array();
           foreach($children as $child){
                if($child->getValue()->hasTag()){
                    if($this === $child){
                        continue;
                    }
                    $array[] = $child;
                }
           }
           return $array;
       }
       return false;
    }
    
    public function getSiblingsByValue($method,$arg){
        $children = $this->parent->getChildren();
        $array = array();
        foreach($children as $child){
             $element = $child->getValue();
             if($element->hasTag()){
                 if($this === $child){
                     continue;
                 } elseif ($element->$method($arg)){
                     $array[] = $child;
                 }
             }
        }
        return $array;
    }
    
    public function hasSiblings(){
       return count($this->getSiblings()) > 1;
    }

    
    public function getSiblingsIndex($tagOnly = true){
        $i = 0;
        foreach($this->parent->getChildren() as $child){
            if($tagOnly){
                if(!$child->getValue()->hasTag()){
                    continue;
                }
            }
            if($child === $this){
                return $i;
            }
            $i++;
        }
        return -1;
    }
}
