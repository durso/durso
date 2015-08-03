<?php

namespace library\mediator;
use library\dom\elements\element;
use library\dom\elements\elementCollection;
use library\tree\node;
use library\tree\branch;


class nodeElement{
    
    public static function children(element $element, $selector = false){
        $children = self::getChildren($element->getNode(), "byValue", $selector);
        return self::nodeToCollection($children);
    }
    public static function siblings(element $element, $selector = false){
        $children = self::getChildren($element->getNode()->getParent(), "byValue", $selector,$element->getNode());
        return self::nodeToCollection($children);
    }
    
     public static function find(element $element, $selector){
        $children = self::bft($element->getNode(),"is",$selector,false);
        return self::nodeToCollection($children);
    }
    
    public static function first(element $element, $selector){
        $children = self::bft($element->getNode(),"is",$selector);
        return self::nodeToCollection($children);
    }
    
    public static function closest(element $element,$selector){
        $parent = self::getAncestors($element->getNode(),$selector,"byValue",true);
        return self::nodeToCollection($parent);
    }
    
    public static function parents($selector = false){
        $parents = self::getAncestors($element->getNode(),$selector,"byValue",false);
        return self::nodeToCollection($parents);
    }
    
    public static function buildSelector(element $element){
        $selector = "";
        $list = self::getAncestors($element->getNode(),"hasId","byMethod",false,true);
        foreach($list as $item){
            $element = $item->getValue();
            if($element->getTag() == 'body'){
                continue;
            }
            if($element->hasId()){
                $selector .= "#".$element->getId()." > ";
            } else {
                $selector .= $element->getTag();
                if($element->hasClass()){
                    $selector .= ".".$element->getClassByIndex(0)." > ";
                } else {
                    $selector .= $element->nthChild()." > ";
                }
            }
        }
        return $selector;
    }
    
    private static function getAncestors($node,$arg,$comp,$single = false, $break = false){
        $list = array();
        while($node->hasParent()){
            $parent = $node->getParent();
            if(self::$comp($parent,$arg)){
                if($single){
                    return array($parent);
                }
                array_unshift($list, $parent);
                if($break){
                    break;
                }
            }elseif($break){
                array_unshift($list, $parent);
            }
            $node = $parent;
        }
        return $list;
    }
    

    
    private static function getChildren($node, $comp, $arg = false, $self = false){
        assert($node->hasChild());
        $children = $node->getChildren();
        $list = array();
        foreach($children as $child){
            if($self){
               if(self::bySelf($child,$self)){
                   continue;
               } 
            } 
            if(self::$comp($child,$arg)){
                $list[] = $child;
            }
        }
        return $list;
    }
    
    public static function childIndex(element $element){
        $i = 0;
        $children = $element->getNode()->getParent()->getChildren();
        $node = $element->getNode();
        foreach($children as $child){
            if(!($child->getValue() instanceof element)){
                    continue;
            }
            if($child === $node){
                return $i;
            }
            $i++;
        }
        return -1;
    }
    
    private static function byValue($node,$arg){
        $element = $node->getValue();
        if($element instanceof element){
            if(!$arg){
                return true;
            }
            return $element->is($arg);
        }
        return false;
    }
    
    private static function byMethod($node,$method){
        return $node->getValue()->$method() || ($node->getValue()->getTag() == 'body');
    }
    
    private static function bySelf($node,$self){
        return $node === $self;
    }
    
    private static function getChildrenByValue($node,$method,$arg){
        assert($node->hasChild());
        $children = $node->getChildren();
        $list = array();
        foreach($children as $child){
            $element = $child->getValue();
            if($element->getValue() instanceof element){
                if($element->$method($arg)){
                    $list[] = $child;
                } 
            }
        }
        return $list;
    }

    private static function bft($node,$method,$arg,$single = true){
        assert(!empty($node->getChildren()));
        $array = array();
        $list = $node->getChildren();
        while(!empty($list)){
            $node = array_shift($list);
            $element = $node->getValue();
            if($element instanceof element){
                if($element->$method($arg)){
                    if($single){
                        return array($node);
                    }
                    $array[] = $node;
                }
            }
            if($node->hasChild()){
                foreach($node->getChildren() as $child){
                    if(!in_array($child,$list,true)){
                        $list[] = $child;
                    }
                }
            }
        }
        return $array;
    }
    
    private static function nodeToComponent($list){
        $array = array();
        foreach($list as $item){
            $array[] = $item->getValue();
        }
        return $array;
    }
    private static function nodeToCollection($list){
        assert(!empty($list));
        if(count($list) == 1){
            return $list[0]->getValue();
        }
        $collection = new elementCollection();
        $components = self::nodeToComponent($list);
        $collection->addElements($components);
        return $collection;
    }
}