<?php
namespace library\tree;
use library\tree\node;
/**
 * Description of leaf
 *
 * @author durso
 */
class leaf extends node {
    public function hasChild(){
	return false;
    }
    public function getChildren(){
	return false;
    }
}
